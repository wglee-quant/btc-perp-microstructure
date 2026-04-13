"""
vortexbar_lab.py
Single-file research script: volume-bar path-based targets with direction/magnitude regression.

Notes:
- Self-contained; no imports from project modules.
- Data: Binance aggTrades (public data zip) only.
- Model: LightGBM regression (default).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Iterable, List

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import numba as nb
except Exception:
    nb = None

EPS = 1e-12
# Features selected for lag expansion.
# Selection criteria: (a) statistically autocorrelated, (b) autocorrelation
# causally connected to VTSR via order-flow persistence, volatility clustering,
# or liquidity state persistence, (c) not already smoothed by a rolling window
# that embeds past information (e.g. cur_vol_vb_bp is excluded for this reason).
LAG_FEATURE_COLS: list[str] = [
    # Flow / toxicity ??order-flow autocorrelation (Cont 2014, Hasbrouck 1991)
    # Institutional order-splitting causes directional pressure to persist across bars.
    "taker_imb1vb",
    "par",
    "twi",
    "large_imb_share",
    "vpin_delta",
    # Price path ??momentum / mean-reversion and trend persistence (Lo-MacKinlay 1988)
    "ret1vb_bp",
    "range1vb_bp",
    "pde",
    "close_pos1vb",
    # Volatility / regime ??regime transition gradient (Ang-Bekaert 2002)
    "vol_ratio",
    # Activity ??market pace cycle persistence
    "dt_sec_z",
    # Microstructure ??liquidity state persistence (Amihud 2002, Madhavan 2000)
    "kyle_lambda",
    "roll_spread",
    # ?? regime-normalised pctrank (lagged) ??
    "ret1vb_bp_pctrank",
    "range1vb_bp_pctrank",
    "kyle_lambda_pctrank",
    "roll_spread_pctrank",
    "vpin_delta_pctrank",
]

# Context columns written to preds.csv alongside predictions.
# Defined at module level so _run_fold_worker can reference them
# without accessing df_feat.
_CONTEXT_COLS: list[str] = [
    "ret1vb_bp",
    "range1vb_bp",
    "taker_imb1vb",
    "volume",
    "vpin_N",
    "vpin_delta",
    "dt_sec",
    "trade_count",
    "tca",
    "par",
    "rev_count_ratio",
    # RT3 trigger diagnostics (NaN when target_mode=vtsr)
    "rt3_on",
    "rt3_off",
    "entry_direction",
    "spread_pct",
    "imb_consistency",
]
BINANCE_DATA_BASE = "https://data.binance.vision/"
MAX_TRAIN_NAN_FRAC = 0.2


class StatusBar:
    def __init__(self, enabled: bool = True, every_sec: float = 2.0, width: int = 120) -> None:
        self.enabled = bool(enabled) and bool(sys.stdout.isatty())
        self.every_sec = float(every_sec)
        self.width = int(width)
        self._last = 0.0
        self._active = False

    def _emit(self, msg: str, newline: bool = False) -> None:
        if not self.enabled:
            if newline:
                print(msg)
            return
        line = ("\r" + msg).ljust(self.width)
        sys.stdout.write(line + ("\n" if newline else ""))
        sys.stdout.flush()
        self._active = not newline

    def update(self, stage: str, msg: str, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last) < self.every_sec:
            return
        self._last = now
        self._emit(f"[{stage}] {msg}", newline=False)

    def done(self, stage: str, msg: str) -> None:
        self._last = time.monotonic()
        self._emit(f"[{stage}] {msg}", newline=True)

    def line(self, msg: str) -> None:
        if self.enabled and self._active:
            sys.stdout.write("\r" + (" " * self.width) + "\r")
            sys.stdout.flush()
            self._active = False
        print(msg)


def parse_date_to_utc_ms(s: str) -> int:
    dt = _dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    return int(dt.timestamp() * 1000)


def ensure_monotonic(ts_ms: np.ndarray, name: str) -> None:
    if len(ts_ms) < 2:
        return
    if not np.all(np.diff(ts_ms) >= 0):
        raise ValueError(f"{name} must be monotonic non-decreasing")


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.asarray(a, dtype=float) / (np.asarray(b, dtype=float) + eps)


def robust_rolling_z(
    x: pd.Series,
    window,
    min_periods: int,
) -> pd.Series:
    med = x.rolling(window, min_periods=min_periods).median()
    mad = (x - med).abs().rolling(window, min_periods=min_periods).median()
    denom = (1.4826 * mad).replace(0, np.nan)
    z = (x - med) / denom
    return z.replace([np.inf, -np.inf], np.nan)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(pd.Series(a[mask]).corr(pd.Series(b[mask]), method="spearman"))


def _sign_acc(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true) & (np.abs(y_true) > eps)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def _rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def _top_k_stats(y_pred: np.ndarray, y_true: np.ndarray, pct: float) -> dict:
    """
    Select top pct fraction of bars by abs(y_pred) (confidence).
    P&L proxy = sign(y_pred) * y_true  (go in predicted direction).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() == 0:
        return {
            "n_selected": 0,
            "cov": 0.0,
            "sign_acc": np.nan,
            "mean_pnl_proxy": np.nan,
            "std_pnl_proxy": np.nan,
            "sharpe_proxy": np.nan,
        }
    yp = y_pred[mask]
    yt = y_true[mask]
    n = len(yp)
    k = max(int(np.floor(n * pct)), 1)
    top_idx = np.argpartition(np.abs(yp), -k)[-k:]
    pnl_proxy = np.sign(yp[top_idx]) * yt[top_idx]
    sign_acc = float(np.mean(np.sign(yp[top_idx]) == np.sign(yt[top_idx])))
    mean_pnl = float(np.nanmean(pnl_proxy))
    std_pnl = float(np.nanstd(pnl_proxy))
    sharpe = float(mean_pnl / std_pnl) if std_pnl > 0 else np.nan
    return {
        "n_selected": int(k),
        "cov": float(k / n),
        "sign_acc": sign_acc,
        "mean_pnl_proxy": mean_pnl,
        "std_pnl_proxy": std_pnl,
        "sharpe_proxy": sharpe,
    }


def _hash_file(path: str, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_checksum_text(text: str) -> tuple[str, str]:
    parts = text.strip().split()
    if not parts:
        raise ValueError("empty checksum file")
    chk = parts[0].strip()
    if len(chk) == 64:
        return "sha256", chk
    if len(chk) == 32:
        return "md5", chk
    raise ValueError(f"unsupported checksum length: {len(chk)}")


def _download_to(path: str, url: str, timeout: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_daily_aggtrades(
    symbol: str,
    date: _dt.date,
    cache_dir: str,
    timeout: float,
    sleep_s: float,
    max_retries: int,
) -> str:
    dstr = date.strftime("%Y-%m-%d")
    fname = f"{symbol}-aggTrades-{dstr}.zip"
    rel = os.path.join("aggTrades", symbol, fname)
    fpath = os.path.join(cache_dir, rel)
    checksum_path = fpath + ".CHECKSUM"
    url = f"{BINANCE_DATA_BASE}data/futures/um/daily/aggTrades/{symbol}/{fname}"
    checksum_url = url + ".CHECKSUM"
    if os.path.exists(fpath) and os.path.exists(checksum_path):
        with open(checksum_path, "r", encoding="utf-8") as f:
            algo, chk = _parse_checksum_text(f.read())
        if _hash_file(fpath, algo) == chk:
            return fpath
    last_err = None
    for _ in range(max_retries):
        try:
            _download_to(checksum_path, checksum_url, timeout)
            with open(checksum_path, "r", encoding="utf-8") as f:
                algo, chk = _parse_checksum_text(f.read())
            _download_to(fpath, url, timeout)
            if _hash_file(fpath, algo) != chk:
                raise ValueError("checksum mismatch")
            return fpath
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def iter_aggtrades_from_zip(
    zip_path: str,
) -> Iterable[tuple[int, float, float, bool, int]]:
    """
    Read one daily aggTrades zip using pandas bulk read for speed.

    Yields (ts_ms, price, qty, is_buyer_maker, agg_id) tuples in
    chronological order. Monotonicity is validated in bulk via numpy.
    Header row (if present) is detected and skipped automatically.
    """
    import re as _re

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            return
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        name = csv_names[0] if csv_names else names[0]
        with zf.open(name, "r") as raw_f:
            peek = io.TextIOWrapper(raw_f, encoding="utf-8", newline="")
            first_line = peek.readline()
            has_header = not bool(
                _re.match(r"^\s*\d+", first_line.lstrip("\ufeff"))
            )

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(name, "r") as raw_f:
            try:
                df = pd.read_csv(
                    raw_f,
                    header=0 if has_header else None,
                    usecols=[0, 1, 2, 5, 6],
                    names=["agg_id", "price", "qty", "ts_ms", "ibm"]
                    if not has_header
                    else None,
                    dtype={
                        0: "int64",
                        1: "float64",
                        2: "float64",
                        5: "int64",
                    },
                )
            except Exception as exc:
                raise ValueError(
                    f"pandas read failed for {zip_path}::{name}"
                ) from exc

    if has_header:
        df.columns = [
            c.strip().lstrip("\ufeff").lower() for c in df.columns
        ]
        col_list = df.columns.tolist()
        rename_map = {
            col_list[0]: "agg_id",
            col_list[1]: "price",
            col_list[2]: "qty",
            col_list[3]: "ts_ms",
            col_list[4]: "ibm",
        }
        df = df.rename(columns=rename_map)

    if df.empty:
        return

    agg_id_arr = df["agg_id"].to_numpy(dtype=np.int64)
    price_arr = df["price"].to_numpy(dtype=np.float64)
    qty_arr = df["qty"].to_numpy(dtype=np.float64)
    ts_ms_arr = df["ts_ms"].to_numpy(dtype=np.int64)
    ibm_col = df["ibm"]
    if ibm_col.dtype == object or ibm_col.dtype.kind in ("U", "S"):
        ibm_arr = (
            ibm_col.astype(str).str.strip().str.lower() == "true"
        ).to_numpy(dtype=bool)
    else:
        ibm_arr = ibm_col.to_numpy(dtype=bool)

    # bulk monotonicity check (replaces per-element loop)
    if len(ts_ms_arr) > 1:
        ts_diff = np.diff(ts_ms_arr)
        if np.any(ts_diff < 0):
            raise ValueError("aggTrades not monotonic by timestamp")
        eq_mask = ts_diff == 0
        if np.any(eq_mask):
            id_diff_at_eq = np.diff(agg_id_arr)[eq_mask]
            if np.any(id_diff_at_eq < 0):
                raise ValueError(
                    "aggTrades not monotonic by agg_id at equal timestamp"
                )

    # yield numpy scalars directly (no int/float/bool conversion)
    n = len(ts_ms_arr)
    for i in range(n):
        yield ts_ms_arr[i], price_arr[i], qty_arr[i], ibm_arr[i], agg_id_arr[i]


def iter_aggtrades_from_zips(
    zip_paths: Iterable[str],
    start_ms: int,
    end_ms: int,
) -> Iterable[tuple[int, float, float, bool, int]]:
    """
    Stream aggTrades from multiple daily zips within [start_ms, end_ms).

    Optimisation: uses _zip_date_ms_range to skip entire zip files
    whose date falls outside the requested range.
    Cross-zip monotonicity is checked at zip boundaries only.
    """
    global_last_ts = None
    global_last_id = None

    for zip_path in zip_paths:
        try:
            zd_start, zd_end = _zip_date_ms_range(zip_path)
            if zd_end <= start_ms or zd_start >= end_ms:
                continue
        except ValueError:
            pass

        for ts_ms, price, qty, is_buyer_maker, agg_id in iter_aggtrades_from_zip(
            zip_path
        ):
            if ts_ms < start_ms:
                continue
            if ts_ms >= end_ms:
                return
            if global_last_ts is not None:
                if ts_ms < global_last_ts or (
                    ts_ms == global_last_ts and agg_id < global_last_id
                ):
                    raise ValueError(
                        "aggTrades not monotonic across zip files"
                    )
            global_last_ts = ts_ms
            global_last_id = agg_id
            yield ts_ms, price, qty, is_buyer_maker, agg_id


def _read_zip_arrays(
    zip_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one daily aggTrades zip and return raw numpy arrays.

    Returns (ts_ms, price, qty, is_buyer_maker) as contiguous arrays.
    Uses binary .npz cache for 10-30x faster repeated reads.
    """
    npz_path = zip_path.rsplit(".", 1)[0] + ".tick.npz"
    try:
        if os.path.exists(npz_path):
            if os.path.getmtime(npz_path) >= os.path.getmtime(zip_path):
                data = np.load(npz_path)
                return (
                    data["ts_ms"],
                    data["price"],
                    data["qty"],
                    data["ibm"],
                )
    except Exception:
        pass

    import re as _re

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=bool),
            )
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        name = csv_names[0] if csv_names else names[0]
        with zf.open(name, "r") as raw_f:
            peek = io.TextIOWrapper(raw_f, encoding="utf-8", newline="")
            first_line = peek.readline()
            has_header = not bool(
                _re.match(r"^\s*\d+", first_line.lstrip("\ufeff"))
            )

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(name, "r") as raw_f:
            df = pd.read_csv(
                raw_f,
                header=0 if has_header else None,
                usecols=[1, 2, 5, 6],
                names=["price", "qty", "ts_ms", "ibm"]
                if not has_header
                else None,
                dtype={1: "float64", 2: "float64", 5: "int64"},
            )

    if has_header:
        df.columns = [
            c.strip().lstrip("\ufeff").lower() for c in df.columns
        ]
        col_list = df.columns.tolist()
        rename_map = {
            col_list[0]: "price",
            col_list[1]: "qty",
            col_list[2]: "ts_ms",
            col_list[3]: "ibm",
        }
        df = df.rename(columns=rename_map)

    if df.empty:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )

    ts_ms = df["ts_ms"].to_numpy(dtype=np.int64)
    price = df["price"].to_numpy(dtype=np.float64)
    qty = df["qty"].to_numpy(dtype=np.float64)
    ibm_col = df["ibm"]
    if ibm_col.dtype == object or ibm_col.dtype.kind in ("U", "S"):
        ibm = (
            ibm_col.astype(str).str.strip().str.lower() == "true"
        ).to_numpy(dtype=bool)
    else:
        ibm = ibm_col.to_numpy(dtype=bool)

    del df

    try:
        np.savez(
            npz_path,
            ts_ms=ts_ms,
            price=price,
            qty=qty,
            ibm=ibm,
        )
    except Exception:
        pass

    return ts_ms, price, qty, ibm


def _zip_date_ms_range(zip_path: str) -> tuple[int, int]:
    """
    Parse date from zip filename and return (day_start_ms, day_end_ms).

    Filename format: {SYMBOL}-aggTrades-{YYYY-MM-DD}.zip
    Returns [00:00:00 UTC, next day 00:00:00 UTC) in epoch ms.
    """
    import re as _re

    basename = os.path.basename(zip_path)
    m = _re.search(r"(\d{4}-\d{2}-\d{2})", basename)
    if not m:
        raise ValueError(f"cannot parse date from zip filename: {basename}")
    d = _dt.datetime.strptime(m.group(1), "%Y-%m-%d").replace(
        tzinfo=_dt.timezone.utc
    )
    day_start_ms = int(d.timestamp() * 1000)
    day_end_ms = day_start_ms + 86_400_000
    return day_start_ms, day_end_ms


def _path_to_day_start_ms(zip_path: str) -> int:
    """Extract day start timestamp (ms) from zip filename."""
    return _zip_date_ms_range(zip_path)[0]


def load_tick_arrays(
    zip_paths: list[str],
    start_ms: int,
    end_ms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load tick data from all zips into numpy arrays.
    Memory-efficient: pre-allocates output, copies chunk-by-chunk,
    frees each chunk immediately to avoid 2× peak memory.

    Returns: (ts_ms, price, qty, is_buyer_maker) filtered to [start_ms, end_ms).
    """
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    total_len = 0

    for zip_path in zip_paths:
        try:
            zd_start, zd_end = _zip_date_ms_range(zip_path)
            if zd_end <= start_ms or zd_start >= end_ms:
                continue
        except ValueError:
            pass

        z_ts, z_price, z_qty, z_ibm = _read_zip_arrays(zip_path)
        if len(z_ts) == 0:
            continue
        mask = (z_ts >= start_ms) & (z_ts < end_ms)
        if not np.any(mask):
            continue
        c_ts = z_ts[mask]
        c_price = z_price[mask]
        c_qty = z_qty[mask]
        c_ibm = z_ibm[mask]
        chunks.append((c_ts, c_price, c_qty, c_ibm))
        total_len += len(c_ts)
        del z_ts, z_price, z_qty, z_ibm

    if total_len == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )

    ts_all = np.empty(total_len, dtype=np.int64)
    price_all = np.empty(total_len, dtype=np.float64)
    qty_all = np.empty(total_len, dtype=np.float64)
    ibm_all = np.empty(total_len, dtype=bool)

    offset = 0
    for i in range(len(chunks)):
        c_ts, c_price, c_qty, c_ibm = chunks[i]
        n = len(c_ts)
        ts_all[offset:offset + n] = c_ts
        price_all[offset:offset + n] = c_price
        qty_all[offset:offset + n] = c_qty
        ibm_all[offset:offset + n] = c_ibm
        offset += n
        chunks[i] = None
    del chunks

    if total_len > 1:
        ts_diff = np.diff(ts_all)
        if np.any(ts_diff < 0):
            raise ValueError("aggTrades not monotonic across zip files")
        del ts_diff

    return ts_all, price_all, qty_all, ibm_all


if nb is not None:
    @nb.njit(cache=True)
    def _build_vb_numba(
        ts_ms_arr: np.ndarray,
        price_arr: np.ndarray,
        qty_arr: np.ndarray,
        ibm_arr: np.ndarray,
        v_target: float,
    ) -> tuple:
        EPS_local = 1e-12
        TINY_EPS = 1e-9
        BIN_LO = -2.0
        BIN_HI = 7.0
        N_BINS = 18
        BIN_SPAN = BIN_HI - BIN_LO
        LOG_N_BINS = np.log(18.0)
        MAX_TRADES_PER_BAR = 100_000

        n_ticks = len(ts_ms_arr)
        if n_ticks == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        max_bars = n_ticks // 500 + 10000
        if max_bars < 1:
            max_bars = 1

        out_ts_first = np.empty(max_bars, dtype=np.int64)
        out_ts_ms = np.empty(max_bars, dtype=np.int64)
        out_open = np.empty(max_bars, dtype=np.float64)
        out_high = np.empty(max_bars, dtype=np.float64)
        out_low = np.empty(max_bars, dtype=np.float64)
        out_close = np.empty(max_bars, dtype=np.float64)
        out_vwap = np.empty(max_bars, dtype=np.float64)
        out_volume = np.empty(max_bars, dtype=np.float64)
        out_trade_count = np.empty(max_bars, dtype=np.int64)
        out_taker_buy_vol = np.empty(max_bars, dtype=np.float64)
        out_taker_sell_vol = np.empty(max_bars, dtype=np.float64)
        out_dt_sec = np.empty(max_bars, dtype=np.float64)
        out_range1vb_bp = np.empty(max_bars, dtype=np.float64)
        out_entropy_norm = np.empty(max_bars, dtype=np.float64)
        out_large_imb_share = np.empty(max_bars, dtype=np.float64)
        out_mid_size_share = np.empty(max_bars, dtype=np.float64)
        out_par = np.empty(max_bars, dtype=np.float64)
        out_rev_count_ratio = np.empty(max_bars, dtype=np.float64)
        out_pde = np.empty(max_bars, dtype=np.float64)
        out_twi = np.empty(max_bars, dtype=np.float64)
        out_roll_spread = np.empty(max_bars, dtype=np.float64)
        out_kyle_lambda = np.empty(max_bars, dtype=np.float64)
        out_burst_cv = np.empty(max_bars, dtype=np.float64)

        bin_counts = np.zeros(N_BINS, dtype=np.int64)
        imb_notional = np.empty(MAX_TRADES_PER_BAR, dtype=np.float64)
        imb_signed_qty = np.empty(MAX_TRADES_PER_BAR, dtype=np.float64)

        bar_count = 0
        active = False

        for i in range(n_ticks):
            ts = ts_ms_arr[i]
            price = price_arr[i]
            qty = qty_arr[i]
            ibm = ibm_arr[i]
            qty_left = qty

            while qty_left > 0.0:
                if not active:
                    cur_ts_first = ts
                    cur_ts_last = ts
                    cur_open = price
                    cur_high = price
                    cur_low = price
                    cur_close = price
                    cur_volume = 0.0
                    cur_notional_sum = 0.0
                    cur_trade_count = 0
                    cur_taker_buy_vol = 0.0
                    cur_taker_sell_vol = 0.0
                    cur_taker_buy_notional = 0.0
                    cur_taker_sell_notional = 0.0
                    cur_path_abs_sum = 0.0
                    cur_has_prev_price = False
                    cur_prev_price = 0.0
                    cur_prev_dp_sign = 0
                    cur_rev_count = 0
                    cur_kl_net_flow = 0.0
                    cur_twi_num = 0.0
                    cur_has_prev_ts = False
                    cur_prev_ts = 0
                    cur_rc_n = 0
                    cur_rc_sum_a = 0.0
                    cur_rc_sum_b = 0.0
                    cur_rc_sum_ab = 0.0
                    cur_rc_prev_dp = 0.0
                    cur_rc_has_prev = False
                    cur_wf_n = 0
                    cur_wf_mean = 0.0
                    cur_wf_M2 = 0.0
                    bin_counts[:] = 0
                    imb_count = 0
                    active = True

                rem = v_target - cur_volume
                if rem <= 0.0:
                    cur_volume = v_target
                    active = False
                    continue

                fill = qty_left if qty_left <= rem else rem

                cur_ts_last = ts
                if price > cur_high:
                    cur_high = price
                if price < cur_low:
                    cur_low = price
                cur_close = price
                cur_volume += fill
                cur_notional_sum += price * fill
                cur_trade_count += 1
                notional = price * fill

                if ibm:
                    cur_taker_sell_vol += fill
                    cur_taker_sell_notional += notional
                    sign = -1.0
                else:
                    cur_taker_buy_vol += fill
                    cur_taker_buy_notional += notional
                    sign = 1.0

                cur_kl_net_flow += sign * fill

                logn = np.log10(max(notional, EPS_local))
                bin_idx = int((logn - BIN_LO) / BIN_SPAN * N_BINS)
                if bin_idx < 0:
                    bin_idx = 0
                if bin_idx >= N_BINS:
                    bin_idx = N_BINS - 1
                bin_counts[bin_idx] += 1

                if imb_count < MAX_TRADES_PER_BAR:
                    imb_notional[imb_count] = notional
                    imb_signed_qty[imb_count] = sign * fill
                    imb_count += 1

                if cur_has_prev_price:
                    dp = price - cur_prev_price
                    cur_path_abs_sum += abs(dp)
                    if dp > 0.0:
                        dp_sign = 1
                    elif dp < 0.0:
                        dp_sign = -1
                    else:
                        dp_sign = 0
                    if cur_prev_dp_sign != 0 and dp_sign != 0 and dp_sign != cur_prev_dp_sign:
                        cur_rev_count += 1
                    cur_prev_dp_sign = dp_sign
                    if cur_rc_has_prev:
                        a = cur_rc_prev_dp
                        b = dp
                        cur_rc_sum_a += a
                        cur_rc_sum_b += b
                        cur_rc_sum_ab += a * b
                        cur_rc_n += 1
                    cur_rc_prev_dp = dp
                    cur_rc_has_prev = True
                cur_has_prev_price = True
                cur_prev_price = price

                if cur_has_prev_ts:
                    dt_ms_val = ts - cur_prev_ts
                    if dt_ms_val < 0:
                        dt_ms_val = 0
                else:
                    dt_ms_val = 0
                w = 1.0 / np.log(2.0 + float(dt_ms_val))
                cur_twi_num += sign * fill * w
                if cur_has_prev_ts and dt_ms_val > 0:
                    iat = float(dt_ms_val)
                    cur_wf_n += 1
                    delta_wf = iat - cur_wf_mean
                    cur_wf_mean += delta_wf / float(cur_wf_n)
                    cur_wf_M2 += delta_wf * (iat - cur_wf_mean)
                cur_prev_ts = ts
                cur_has_prev_ts = True

                qty_left -= fill

                if cur_volume >= v_target - TINY_EPS:
                    cur_volume = v_target

                    vwap = cur_notional_sum / max(cur_volume, EPS_local)
                    dt_sec = float(cur_ts_last - cur_ts_first) / 1000.0
                    range1vb_bp = (cur_high - cur_low) / (cur_close + EPS_local) * 1e4
                    net_disp = abs(cur_close - cur_open)
                    pde = net_disp / (cur_path_abs_sum + EPS_local)
                    if pde < 0.0:
                        pde = 0.0
                    if pde > 1.0:
                        pde = 1.0

                    tc = cur_trade_count
                    if tc >= 5:
                        total_bins = 0
                        for j in range(N_BINS):
                            total_bins += bin_counts[j]
                        if total_bins == 0:
                            entropy_norm = np.nan
                        else:
                            inv = 1.0 / float(total_bins)
                            h = 0.0
                            for j in range(N_BINS):
                                c = bin_counts[j]
                                if c > 0:
                                    p = float(c) * inv
                                    h -= p * np.log(p + EPS_local)
                            entropy_norm = h / max(LOG_N_BINS, EPS_local)
                            if not np.isfinite(entropy_norm):
                                entropy_norm = np.nan
                    else:
                        entropy_norm = np.nan

                    if imb_count > 0:
                        k = imb_count // 20
                        if k < 1:
                            k = 1
                        sorted_idx = np.argsort(imb_notional[:imb_count])
                        large_imb = 0.0
                        for j in range(imb_count - k, imb_count):
                            large_imb += imb_signed_qty[sorted_idx[j]]
                        large_imb_share = large_imb / (cur_volume + EPS_local)
                    else:
                        large_imb_share = 0.0

                    mid_bin_count = 0
                    for j in range(7, 12):
                        mid_bin_count += bin_counts[j]
                    mid_size_share = float(mid_bin_count) / max(float(tc), 1.0)

                    buy_vol = cur_taker_buy_vol
                    sell_vol = cur_taker_sell_vol
                    if buy_vol > 0.0:
                        avg_buy_px = cur_taker_buy_notional / buy_vol
                    else:
                        avg_buy_px = cur_close
                    if sell_vol > 0.0:
                        avg_sell_px = cur_taker_sell_notional / sell_vol
                    else:
                        avg_sell_px = cur_close
                    total_abs = cur_volume
                    par = (buy_vol * avg_buy_px - sell_vol * avg_sell_px) / (
                        total_abs * cur_close + EPS_local
                    )

                    rev_count_ratio = float(cur_rev_count) / max(float(tc), 1.0)

                    if cur_rc_n >= 2:
                        cov_val = (
                            cur_rc_sum_ab
                            - cur_rc_sum_a * cur_rc_sum_b / float(cur_rc_n)
                        ) / float(cur_rc_n - 1)
                        neg_cov = -cov_val
                        if neg_cov < 0.0:
                            neg_cov = 0.0
                        roll_spread = 2.0 * np.sqrt(neg_cov)
                        if not np.isfinite(roll_spread):
                            roll_spread = np.nan
                    elif cur_rc_n == 1:
                        roll_spread = 0.0
                    else:
                        roll_spread = np.nan

                    kyle_lambda = cur_path_abs_sum / (abs(cur_kl_net_flow) + EPS_local)

                    if cur_wf_n >= 2 and cur_wf_mean > 0.0:
                        variance = cur_wf_M2 / float(cur_wf_n - 1)
                        if variance < 0.0:
                            variance = 0.0
                        burst_cv = np.sqrt(variance) / cur_wf_mean
                        if not np.isfinite(burst_cv):
                            burst_cv = np.nan
                    elif cur_wf_n >= 1:
                        burst_cv = 0.0
                    else:
                        burst_cv = np.nan

                    twi = cur_twi_num / max(v_target, EPS_local)
                    if twi < -2.0:
                        twi = -2.0
                    if twi > 2.0:
                        twi = 2.0

                    out_ts_first[bar_count] = cur_ts_first
                    out_ts_ms[bar_count] = cur_ts_last
                    out_open[bar_count] = cur_open
                    out_high[bar_count] = cur_high
                    out_low[bar_count] = cur_low
                    out_close[bar_count] = cur_close
                    out_vwap[bar_count] = vwap
                    out_volume[bar_count] = cur_volume
                    out_trade_count[bar_count] = tc
                    out_taker_buy_vol[bar_count] = cur_taker_buy_vol
                    out_taker_sell_vol[bar_count] = cur_taker_sell_vol
                    out_dt_sec[bar_count] = dt_sec
                    out_range1vb_bp[bar_count] = range1vb_bp
                    out_entropy_norm[bar_count] = entropy_norm
                    out_large_imb_share[bar_count] = large_imb_share
                    out_mid_size_share[bar_count] = mid_size_share
                    out_par[bar_count] = par
                    out_rev_count_ratio[bar_count] = rev_count_ratio
                    out_pde[bar_count] = pde
                    out_twi[bar_count] = twi
                    out_roll_spread[bar_count] = roll_spread
                    out_kyle_lambda[bar_count] = kyle_lambda
                    out_burst_cv[bar_count] = burst_cv

                    bar_count += 1
                    active = False

        return (
            out_ts_first[:bar_count],
            out_ts_ms[:bar_count],
            out_open[:bar_count],
            out_high[:bar_count],
            out_low[:bar_count],
            out_close[:bar_count],
            out_vwap[:bar_count],
            out_volume[:bar_count],
            out_trade_count[:bar_count],
            out_taker_buy_vol[:bar_count],
            out_taker_sell_vol[:bar_count],
            out_dt_sec[:bar_count],
            out_range1vb_bp[:bar_count],
            out_entropy_norm[:bar_count],
            out_large_imb_share[:bar_count],
            out_mid_size_share[:bar_count],
            out_par[:bar_count],
            out_rev_count_ratio[:bar_count],
            out_pde[:bar_count],
            out_twi[:bar_count],
            out_roll_spread[:bar_count],
            out_kyle_lambda[:bar_count],
            out_burst_cv[:bar_count],
        )
else:
    def _build_vb_numba(
        ts_ms_arr: np.ndarray,
        price_arr: np.ndarray,
        qty_arr: np.ndarray,
        ibm_arr: np.ndarray,
        v_target: float,
    ) -> tuple:
        raise RuntimeError("numba not installed; required for build_volume_bars")


def build_volume_bars(
    ts_ms_arr: np.ndarray,
    price_arr: np.ndarray,
    qty_arr: np.ndarray,
    ibm_arr: np.ndarray,
    v_target: float,
) -> pd.DataFrame:
    """
    Build volume bars from pre-loaded tick arrays.
    Delegates to Numba-compiled _build_vb_numba for speed.
    """
    result = _build_vb_numba(ts_ms_arr, price_arr, qty_arr, ibm_arr, v_target)

    (
        ts_first,
        ts_ms,
        open_,
        high,
        low,
        close,
        vwap,
        volume,
        trade_count,
        taker_buy_vol,
        taker_sell_vol,
        dt_sec,
        range1vb_bp,
        entropy_norm,
        large_imb_share,
        mid_size_share,
        par,
        rev_count_ratio,
        pde,
        twi,
        roll_spread,
        kyle_lambda,
        burst_cv,
    ) = result

    return pd.DataFrame(
        {
            "ts_first": ts_first,
            "ts_ms": ts_ms,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vwap": vwap,
            "volume": volume,
            "trade_count": trade_count,
            "taker_buy_vol": taker_buy_vol,
            "taker_sell_vol": taker_sell_vol,
            "dt_sec": dt_sec,
            "range1vb_bp": range1vb_bp,
            "entropy_norm": entropy_norm,
            "large_imb_share": large_imb_share,
            "mid_size_share": mid_size_share,
            "par": par,
            "rev_count_ratio": rev_count_ratio,
            "pde": pde,
            "twi": twi,
            "roll_spread": roll_spread,
            "kyle_lambda": kyle_lambda,
            "burst_cv": burst_cv,
        }
    )


def build_volume_bars_chunked(
    all_paths: list[str],
    start_ms: int,
    end_ms: int,
    v_target: float,
    chunk_days: int = 60,
    overlap_days: int = 3,
) -> pd.DataFrame:
    """
    Build volume bars in chunks to avoid OOM on large datasets.

    Splits zip paths into overlapping chunks of ~chunk_days,
    builds VBs per chunk, keeps only bars in the non-overlapping core,
    then concatenates.
    """
    n = len(all_paths)
    if n == 0:
        return pd.DataFrame()

    if n <= chunk_days + 2 * overlap_days:
        ts, price, qty, ibm = load_tick_arrays(all_paths, start_ms, end_ms)
        df = build_volume_bars(ts, price, qty, ibm, v_target)
        del ts, price, qty, ibm
        return df

    all_dfs: list[pd.DataFrame] = []

    for chunk_start in range(0, n, chunk_days):
        chunk_end = min(chunk_start + chunk_days, n)

        load_start = max(0, chunk_start - overlap_days)
        load_end = min(n, chunk_end + overlap_days)

        batch_paths = all_paths[load_start:load_end]
        ts, price, qty, ibm = load_tick_arrays(batch_paths, start_ms, end_ms)

        if len(ts) == 0:
            del ts, price, qty, ibm
            continue

        df_chunk = build_volume_bars(ts, price, qty, ibm, v_target)
        del ts, price, qty, ibm

        if df_chunk.empty:
            continue

        if chunk_start == 0:
            core_start = start_ms
        else:
            core_start = _path_to_day_start_ms(all_paths[chunk_start])

        if chunk_end >= n:
            core_end = end_ms
        else:
            core_end = _path_to_day_start_ms(all_paths[chunk_end])

        ts_ms_arr = df_chunk["ts_ms"].to_numpy()
        mask = (ts_ms_arr >= core_start) & (ts_ms_arr < core_end)
        df_chunk = df_chunk.loc[mask].copy()

        if not df_chunk.empty:
            all_dfs.append(df_chunk)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def aggregate_eth_per_vb(
    vb_ts_first: np.ndarray,
    vb_ts_last: np.ndarray,
    eth_zip_paths: list[str],
    start_ms: int,
    end_ms: int,
) -> dict[str, np.ndarray]:
    """
    Aggregate ETH ticks within each BTC VB window [ts_first, ts_last].
    Streaming: processes one zip (one day) at a time to avoid OOM.

    Returns dict with keys:
      eth_taker_imb, eth_ret_bp, eth_kyle_lambda, eth_volume
    """
    n_bars = len(vb_ts_first)
    EPS_local = 1e-12

    acc_buy_vol = np.zeros(n_bars, dtype=np.float64)
    acc_sell_vol = np.zeros(n_bars, dtype=np.float64)
    acc_first_price = np.full(n_bars, np.nan, dtype=np.float64)
    acc_last_price = np.full(n_bars, np.nan, dtype=np.float64)
    acc_path_abs_sum = np.zeros(n_bars, dtype=np.float64)
    acc_net_flow = np.zeros(n_bars, dtype=np.float64)
    acc_prev_price = np.full(n_bars, np.nan, dtype=np.float64)
    acc_has_data = np.zeros(n_bars, dtype=np.bool_)

    for zip_path in eth_zip_paths:
        try:
            zd_start, zd_end = _zip_date_ms_range(zip_path)
            if zd_end <= start_ms or zd_start >= end_ms:
                continue
        except ValueError:
            pass

        z_ts, z_price, z_qty, z_ibm = _read_zip_arrays(zip_path)
        if len(z_ts) == 0:
            continue
        mask = (z_ts >= start_ms) & (z_ts < end_ms)
        if not np.any(mask):
            continue
        z_ts = z_ts[mask]
        z_price = z_price[mask]
        z_qty = z_qty[mask]
        z_ibm = z_ibm[mask]

        day_start = int(z_ts[0])
        day_end = int(z_ts[-1])

        vb_overlap = np.where(
            (vb_ts_last >= day_start) & (vb_ts_first <= day_end)
        )[0]

        for i in vb_overlap:
            left = int(np.searchsorted(z_ts, vb_ts_first[i], side="left"))
            right = int(np.searchsorted(z_ts, vb_ts_last[i], side="right"))
            if left >= right:
                continue

            p = z_price[left:right]
            q = z_qty[left:right]
            ibm = z_ibm[left:right]

            buy_mask = ~ibm
            sell_mask = ibm
            acc_buy_vol[i] += float(q[buy_mask].sum())
            acc_sell_vol[i] += float(q[sell_mask].sum())

            if np.isnan(acc_first_price[i]):
                acc_first_price[i] = float(p[0])
            acc_last_price[i] = float(p[-1])

            if not np.isnan(acc_prev_price[i]):
                acc_path_abs_sum[i] += abs(float(p[0]) - acc_prev_price[i])
            if len(p) > 1:
                acc_path_abs_sum[i] += float(np.abs(np.diff(p)).sum())
            acc_prev_price[i] = float(p[-1])

            signed_flow = np.where(ibm, -q, q)
            acc_net_flow[i] += float(signed_flow.sum())

            acc_has_data[i] = True

        del z_ts, z_price, z_qty, z_ibm

    out_eth_taker_imb = np.full(n_bars, np.nan, dtype=np.float64)
    out_eth_ret_bp = np.full(n_bars, np.nan, dtype=np.float64)
    out_eth_kyle_lambda = np.full(n_bars, np.nan, dtype=np.float64)
    out_eth_volume = np.full(n_bars, np.nan, dtype=np.float64)

    for i in range(n_bars):
        if not acc_has_data[i]:
            continue
        total = acc_buy_vol[i] + acc_sell_vol[i]
        if total > EPS_local:
            out_eth_taker_imb[i] = (acc_buy_vol[i] - acc_sell_vol[i]) / total
        else:
            out_eth_taker_imb[i] = 0.0

        if acc_first_price[i] > EPS_local:
            out_eth_ret_bp[i] = (
                (acc_last_price[i] - acc_first_price[i])
                / acc_first_price[i]
                * 1e4
            )
        else:
            out_eth_ret_bp[i] = 0.0

        out_eth_kyle_lambda[i] = acc_path_abs_sum[i] / (
            abs(acc_net_flow[i]) + EPS_local
        )
        out_eth_volume[i] = total

    return {
        "eth_taker_imb": out_eth_taker_imb,
        "eth_ret_bp": out_eth_ret_bp,
        "eth_kyle_lambda": out_eth_kyle_lambda,
        "eth_volume": out_eth_volume,
    }


def compute_vtsr_target(
    df: pd.DataFrame,
    horizon: int,
    vol_ewma_span: int,
    eps: float = EPS,
) -> pd.Series:
    """
    VTSR: Vol-Time Scaled Return.

    y[t] = R_H[t] / ( sigma_ewma[t] * sqrt(dt_H[t] / dt_avg) + eps )

    where:
      R_H[t]      = log(close[t+H] / close[t])
      sigma_ewma  = EWMA(|bar log return|, span=vol_ewma_span), causal
      dt_H[t]     = sum(dt_sec[t+1], ..., dt_sec[t+H])  (forward sum)
      dt_avg      = nanmedian(dt_sec) over all bars (global constant)
    """
    H = int(horizon)
    closes = df["close"].astype(float)
    ret_log = np.log(closes.clip(lower=eps)).diff(1)

    # Forward log return over H bars (causal: only uses future close, no future feature)
    R_H = np.log(closes.shift(-H).clip(lower=eps)) - np.log(closes.clip(lower=eps))

    # Causal EWMA volatility estimate (adjust=False ensures causality)
    sigma_ewma = ret_log.abs().ewm(span=int(vol_ewma_span), adjust=False).mean()

    # Forward time sum: dt_H[t] = sum(dt_sec[t+1], ..., dt_sec[t+H])
    dt_sec = df["dt_sec"].astype(float)
    dt_H = pd.Series(0.0, index=df.index, dtype=float)
    for j in range(1, H + 1):
        dt_H += dt_sec.shift(-j).fillna(0.0)

    # Global median bar duration (scalar constant; dimensionless normalization)
    dt_avg = float(np.nanmedian(dt_sec.to_numpy()))

    denom = sigma_ewma * np.sqrt(dt_H / (dt_avg + eps)) + eps
    y = R_H / denom

    # NaN the last H rows: target requires future bars not yet available
    if H > 0:
        y.iloc[-H:] = np.nan

    y.name = "y_vtsr"
    return y


def compute_rt3_trigger(
    df: pd.DataFrame,
    spread_window: int,
    spread_pct_on: float,
    spread_pct_off: float,
    K: int,
    imb_consistency_on: float,
    imb_consistency_off: float,
    off_drop_ratio: float = 0.30,
    off_peak_window: int = 20,
    off_confirm_bars: int = 2,
    eps: float = EPS,
) -> pd.DataFrame:
    """
    RT3 trigger: regime-adaptive spread shock + directional imbalance.

    Causal design: all computations use only past data.
    Based on Glosten-Milgrom (1985) adverse selection detection.

    Returns a DataFrame with columns:
      rt3_on          : bool, trigger active at bar t
      rt3_off         : bool, exit condition met at bar t
      entry_direction : float, +1 (buy) or -1 (sell), valid when rt3_on
      spread_pct      : float, causal rolling percentile of roll_spread
      imb_consistency : float, taker_imb directional consistency over K bars
    """
    spread_window = int(spread_window)
    K = int(K)
    minp_spread = max(3, spread_window // 4)
    minp_K = max(2, K // 2)

    # ── Step 1: causal percentile rank of roll_spread ────────────────
    # rolling().rank(pct=True) references only past spread_window bars.
    # Handles NaN in roll_spread gracefully (NaN rows excluded from rank).
    roll_spread = df["roll_spread"]
    spread_pct = roll_spread.rolling(
        spread_window, min_periods=minp_spread
    ).rank(pct=True)

    # ── Step 2: spread shock detection ──────────────────────────────
    # spread_shock is True when current bar's spread is in top
    # (1 - spread_pct_on) of recent distribution.
    spread_shock = spread_pct >= spread_pct_on

    # spread_shock_recent: was there a shock within the last K bars?
    # rolling max over K bars; True if any bar in window had a shock.
    spread_shock_recent = spread_shock.rolling(
        K, min_periods=minp_K
    ).max().fillna(0).astype(bool)

    # ── Step 3: taker_imb directional consistency over K bars ───────
    # imb_consistency = |sum(sign(imb) over K bars)| / K
    # = 1.0 if all K bars have the same direction,
    # = 0.0 if perfectly alternating.
    # Uses shift(1) so current bar is not included (causal).
    imb_sign = np.sign(df["taker_imb1vb"])
    imb_consistency = (
        imb_sign.shift(1)
        .rolling(K, min_periods=minp_K)
        .sum()
        .abs()
        / K
    )

    # ── Step 4: RT3 ON ───────────────────────────────────────────────
    # Spread shock must have occurred in the last K bars (prior to now),
    # AND current taker_imb is directionally consistent.
    rt3_on = spread_shock_recent & (imb_consistency >= imb_consistency_on)

    # ── Step 5: RT3 OFF — peak-relative drop with N-bar confirmation ─
    #
    # Rationale: the old absolute-threshold exit (spread_pct < 0.40) fired
    # too late (median spread_pct at exit = 0.22) and the imb_consistency
    # branch never fired (imb_consistency was always 1.0 with K=4).
    #
    # New definition: exit when spread has fallen enough from its
    # event-peak AND that condition holds for rt3_off_confirm_bars
    # consecutive bars.  This is relative to the event's own intensity,
    # so a weak spread event and a strong one are treated proportionally.
    #
    # peak_spread[t] = rolling max of spread_pct over the last
    #                  rt3_off_peak_window bars (causal).
    # exit_raw[t]    = True when spread_pct[t] < peak_spread[t] * (1 - drop_ratio)
    # rt3_off[t]     = True when exit_raw has been True for
    #                  rt3_off_confirm_bars consecutive bars.
    #
    # Parameters (hard-coded defaults, exposed as function arguments below):
    #   drop_ratio             = 0.30  (peak must fall 30 % to trigger exit)
    #   rt3_off_peak_window    = 20    (bars to look back for peak spread)
    #   rt3_off_confirm_bars   = 2     (consecutive bars required)
    #
    # Causality: rolling max looks only backward; no future leakage.

    drop_ratio = off_drop_ratio
    rt3_off_peak_window = off_peak_window
    rt3_off_confirm_bars = off_confirm_bars

    # Causal rolling peak of spread_pct over the last rt3_off_peak_window bars
    peak_spread = spread_pct.rolling(
        rt3_off_peak_window, min_periods=1
    ).max()

    # Raw exit signal: spread fell >= drop_ratio from peak
    exit_raw = spread_pct < peak_spread * (1.0 - drop_ratio)

    # Confirmed exit: exit_raw must be True for confirm_bars consecutive bars.
    # rolling(n).min() == 1 iff all n values are True (1).
    if rt3_off_confirm_bars > 1:
        rt3_off = (
            exit_raw.astype(float)
            .rolling(rt3_off_confirm_bars, min_periods=rt3_off_confirm_bars)
            .min()
            .fillna(0)
            .astype(bool)
        )
    else:
        rt3_off = exit_raw

    # ── Step 6: entry direction ─────────────────────────────────────
    # Direction of the dominant taker_imb over K bars.
    entry_direction = np.sign(
        imb_sign.shift(1)
        .rolling(K, min_periods=minp_K)
        .sum()
    )
    entry_direction = entry_direction.where(rt3_on, other=0.0)

    return pd.DataFrame(
        {
            "rt3_on": rt3_on.astype(bool),
            "rt3_off": rt3_off.astype(bool),
            "entry_direction": entry_direction.astype(float),
            "spread_pct": spread_pct.astype(float),
            "imb_consistency": imb_consistency.astype(float),
        },
        index=df.index,
    )


def compute_event_target(
    df: pd.DataFrame,
    rt3: pd.DataFrame,
    vol_ewma_span: int,
    hmax: int,
    eps: float = EPS,
) -> pd.Series:
    """
    Event-based target: VTSR normalization with market-determined duration.

    y_event[t] = log(close[t_exit] / close[t])
                 / (sigma_ewma[t] * sqrt(duration / dt_avg) + eps)

    t_exit is the first bar >= t+1 where rt3_off is True,
    capped at t + hmax. If no exit found within hmax, y_event[t] = NaN.
    If rt3_on[t] is False, y_event[t] = NaN.

    Causality:
      - sigma_ewma uses only data up to and including bar t.
      - t_exit uses only forward bars (future close/trigger state),
        which is legitimate as a label (not a feature).
      - No future data leaks into features.

    Complexity: O(n) via backward-scan precomputation of next_off[].
    """
    closes = df["close"].astype(float)
    ret_log = np.log(closes.clip(lower=eps)).diff(1)
    sigma_ewma = ret_log.abs().ewm(
        span=int(vol_ewma_span), adjust=False
    ).mean()
    dt_avg = float(np.nanmedian(df["dt_sec"].astype(float).to_numpy()))

    rt3_on_arr  = rt3["rt3_on"].to_numpy(dtype=bool)
    rt3_off_arr = rt3["rt3_off"].to_numpy(dtype=bool)
    close_arr   = closes.to_numpy(dtype=float)
    sigma_arr   = sigma_ewma.to_numpy(dtype=float)

    n = len(df)
    hmax_int = int(hmax)

    # ── backward scan: next_off[i] = smallest j >= i where rt3_off ──
    # If no such j exists, next_off[i] = n (sentinel).
    next_off = np.full(n, n, dtype=np.int64)
    if n > 0 and rt3_off_arr[n - 1]:
        next_off[n - 1] = n - 1
    for i in range(n - 2, -1, -1):
        if rt3_off_arr[i]:
            next_off[i] = i
        else:
            next_off[i] = next_off[i + 1]

    y = np.full(n, np.nan, dtype=float)

    for t in range(n):
        if not rt3_on_arr[t]:
            continue
        if not np.isfinite(close_arr[t]) or close_arr[t] <= 0:
            continue
        # First rt3_off strictly after t
        search_start = t + 1
        if search_start >= n:
            continue
        t_exit = int(next_off[search_start])
        # Must be within [t+1, t+hmax]
        if t_exit >= n or t_exit > t + hmax_int:
            continue
        if not np.isfinite(close_arr[t_exit]) or close_arr[t_exit] <= 0:
            continue
        R = math.log(close_arr[t_exit] / close_arr[t])
        duration = t_exit - t
        sigma = float(sigma_arr[t])
        denom = sigma * math.sqrt(duration / (dt_avg + eps)) + eps
        y[t] = R / denom

    return pd.Series(y, index=df.index, name="y_event")


def compute_alpha_target(
    df: pd.DataFrame,
    ca_horizon: int,
    vol_ewma_span: int,  # kept for API compat, unused
    eps: float = EPS,
) -> pd.Series:
    """
    Continuous alpha target: pure Efficiency Ratio.

    alpha_target[t] = ER[t] = net_move[t] / path_length[t]

    where:
      net_move[t]    = |close[t+H] - close[t]|
      path_length[t] = Σ|close[t+i+1] - close[t+i]| for i=0..H-1
      ER ∈ [0, 1]: 1 = straight-line move, 0 = choppy/mean-reverting
    """
    H = int(ca_horizon)
    closes = df["close"].astype(float).to_numpy()
    n = len(closes)

    net_move = np.full(n, np.nan, dtype=float)
    if n > H:
        net_move[: n - H] = np.abs(closes[H:] - closes[: n - H])

    abs_diff = np.abs(np.diff(closes, prepend=np.nan))
    cum_abs = np.nancumsum(abs_diff)
    path_length = np.full(n, np.nan, dtype=float)
    if n > H:
        path_length[: n - H] = cum_abs[H:] - cum_abs[: n - H]

    alpha_target = np.where(
        path_length > eps,
        np.minimum(net_move / path_length, 1.0),
        0.0,
    )
    if H > 0:
        alpha_target[-H:] = np.nan

    return pd.Series(alpha_target, index=df.index, name="alpha_target")


def compute_vwap_entry_target(
    df: pd.DataFrame,
    ca_horizon: int,
    vol_ewma_span: int,
    eps: float = EPS,
) -> pd.Series:
    """
    Entry target: vol-normalized VWAP-to-VWAP return.

    y_entry[t] = (VWAP_fwd[t] - VWAP[t]) / (VWAP[t] * sigma_ewma[t])

    where:
      VWAP[t]     = volume-weighted average price of bar t
      VWAP_fwd[t] = integrated VWAP over bars t+1 to t+H
                   = Σ(notional[i]) / Σ(volume[i]) for i in t+1..t+H
      sigma_ewma  = EWMA(|bar log return|, span=vol_ewma_span), causal
    """
    H = int(ca_horizon)
    vwap_bar = df["vwap"].astype(float).to_numpy()
    volume = df["volume"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    n = len(df)

    notional = vwap_bar * volume
    cum_notional = np.nancumsum(notional)
    cum_volume = np.nancumsum(volume)

    fwd_notional = np.full(n, np.nan, dtype=float)
    fwd_volume = np.full(n, np.nan, dtype=float)
    if n > H:
        fwd_notional[: n - H] = cum_notional[H:] - cum_notional[: n - H]
        fwd_volume[: n - H] = cum_volume[H:] - cum_volume[: n - H]

    vwap_fwd = np.where(
        fwd_volume > eps,
        fwd_notional / fwd_volume,
        np.nan,
    )

    ret_log = np.diff(np.log(np.maximum(closes, eps)), prepend=np.nan)
    ret_log_s = pd.Series(ret_log)
    sigma_ewma = (
        ret_log_s.abs().ewm(span=int(vol_ewma_span), adjust=False).mean().to_numpy()
    )

    denom = vwap_bar * np.maximum(sigma_ewma, eps)
    y_entry = np.where(
        denom > eps,
        (vwap_fwd - vwap_bar) / denom,
        np.nan,
    )
    if H > 0:
        y_entry[-H:] = np.nan

    return pd.Series(y_entry, index=df.index, name="y_entry")


def _make_feature_df(
    df_vb: pd.DataFrame,
    feat_scale: int,
    n_lags: int = 0,
    pctrank_window: int = 500,
) -> pd.DataFrame:
    df = df_vb.copy().sort_values("ts_ms").reset_index(drop=True)
    ensure_monotonic(df["ts_ms"].to_numpy(), "volume bars ts_ms")

    df["ret1vb_bp"] = safe_div(df["close"] - df["close"].shift(1), df["close"].shift(1)) * 1e4
    df["retHvb_bp"] = safe_div(
        df["close"] - df["close"].shift(feat_scale),
        df["close"].shift(feat_scale),
    ) * 1e4
    df["range1vb_bp"] = safe_div(df["high"] - df["low"], df["close"]) * 1e4
    rng = df["high"] - df["low"]
    close_pos = pd.Series(0.0, index=df.index, dtype=float)
    mask = rng.abs() >= EPS
    close_pos.loc[mask] = (df.loc[mask, "close"] - df.loc[mask, "low"]) / rng.loc[mask] - 0.5
    df["close_pos1vb"] = close_pos
    taker_total = df["taker_buy_vol"] + df["taker_sell_vol"]
    df["taker_imb1vb"] = safe_div(df["taker_buy_vol"] - df["taker_sell_vol"], taker_total + EPS)

    window_v = max(10, int(4 * feat_scale))
    minp_v = max(5, window_v // 4)
    df["dt_sec_z"] = robust_rolling_z(np.log1p(df["dt_sec"].clip(lower=0.0)), window=window_v, min_periods=minp_v)
    df["trade_count_z"] = robust_rolling_z(
        np.log1p(df["trade_count"].clip(lower=0.0)), window=window_v, min_periods=minp_v
    )
    tc_log = np.log(df["trade_count"].clip(lower=EPS))
    tc_mean = df["trade_count"].rolling(
        feat_scale, min_periods=max(3, feat_scale // 4)
    ).mean().clip(lower=EPS)
    df["tca"] = tc_log - np.log(tc_mean)

    ret_log = np.log(df["close"].clip(lower=EPS)).diff(1)
    ret_bp = ret_log * 1e4
    vol_win = max(10, feat_scale)
    minp = max(3, vol_win // 4)
    vol_ret = np.sqrt((ret_bp ** 2).rolling(vol_win, min_periods=minp).mean())
    range_mean = df["range1vb_bp"].rolling(vol_win, min_periods=minp).mean()
    df["cur_vol_vb_bp"] = np.maximum(vol_ret, range_mean)

    # close_vs_vwap: close position relative to vwap, scaled by bar range
    hl = (df["high"] - df["low"]).clip(lower=EPS)
    df["close_vs_vwap"] = (df["close"] - df["vwap"]) / hl

    # vol_ratio: short-term / long-term volatility (regime change detector)
    vol_short_N = max(5, feat_scale // 2)
    alpha_s = 2.0 / (vol_short_N + 1.0)
    alpha_l = 2.0 / (vol_win + 1.0)
    vol_short_ewm = ret_bp.abs().ewm(alpha=alpha_s, adjust=False).mean()
    vol_long_ewm = ret_bp.abs().ewm(alpha=alpha_l, adjust=False).mean()
    df["vol_ratio"] = safe_div(vol_short_ewm, vol_long_ewm + EPS)

    # ── Cross-pair derived features ───────────────────────────────
    if "eth_taker_imb" in df.columns:
        df["imb_divergence"] = df["eth_taker_imb"] - df["taker_imb1vb"]
    if "eth_ret_bp" in df.columns:
        df["ret_divergence"] = df["eth_ret_bp"] - df["ret1vb_bp"]

    # ?? Rolling percentile rank: regime-normalised features ?????????????
    # ?? ???(bp, ??? ??) ??? ?? 500? ?? ?? ??? ??.
    # causal rolling window? ????? ?? ?? ?? ??.
    # raw ??? ???? ??? ??? / ??? ??? ??? ? ?? ??.
    _pctrank_window = pctrank_window
    _pctrank_minp = max(10, _pctrank_window // 10)

    _PCTRANK_COLS = [
        "ret1vb_bp",
        "retHvb_bp",
        "range1vb_bp",
        "cur_vol_vb_bp",
        "kyle_lambda",
        "roll_spread",
        "vpin_delta",
    ]
    for _col in _PCTRANK_COLS:
        if _col in df.columns:
            df[f"{_col}_pctrank"] = (
                df[_col]
                .rolling(_pctrank_window, min_periods=_pctrank_minp)
                .rank(pct=True)
            )


    # Lag expansion — only for features in LAG_FEATURE_COLS, only when n_lags > 0.
    # shift(k) for k in 1..n_lags; NaN rows at the head are handled by SimpleImputer
    # downstream. n_lags=0 produces identical output to the pre-lag version.
    if n_lags > 0:
        for col in LAG_FEATURE_COLS:
            if col not in df.columns:
                continue
            for lag in range(1, n_lags + 1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def _build_feature_cols(n_lags: int = 0) -> List[str]:
    cols = [
        # price path group
        "ret1vb_bp",
        "retHvb_bp",
        "range1vb_bp",
        "close_pos1vb",
        "close_vs_vwap",
        "pde",
        # volatility / regime group
        "cur_vol_vb_bp",
        "vol_ratio",
        # flow / toxicity group
        "taker_imb1vb",
        "par",
        "twi",
        "large_imb_share",
        "vpin_N",
        "vpin_delta",
        # microstructure group
        "rev_count_ratio",
        "entropy_norm",
        "kyle_lambda",
        "roll_spread",
        # activity group
        "dt_sec_z",
        "trade_count_z",
        "tca",
        "burst_cv",
        # ?? regime-normalised percentile rank ??
        "ret1vb_bp_pctrank",
        "retHvb_bp_pctrank",
        "range1vb_bp_pctrank",
        "cur_vol_vb_bp_pctrank",
        "kyle_lambda_pctrank",
        "roll_spread_pctrank",
        "vpin_delta_pctrank",
    ]
    # Append lag column names in the same order as they are generated in
    # _make_feature_df so that X matrix columns align exactly.
    for col in LAG_FEATURE_COLS:
        for lag in range(1, n_lags + 1):
            cols.append(f"{col}_lag{lag}")
    return cols


def _nan_fraction_rows(X: np.ndarray) -> float:
    if X.size == 0:
        return 1.0
    return float(np.mean(np.isnan(X).any(axis=1)))


def _fill_all_nan_cols(X_train: np.ndarray, X_oos: np.ndarray) -> tuple[np.ndarray, np.ndarray, List[int]]:
    all_nan_cols = []
    for i in range(X_train.shape[1]):
        if not np.isfinite(X_train[:, i]).any():
            all_nan_cols.append(i)
    if all_nan_cols:
        X_train = X_train.copy()
        X_oos = X_oos.copy()
        X_train[:, all_nan_cols] = 0.0
        X_oos[:, all_nan_cols] = 0.0
    return X_train, X_oos, all_nan_cols


def _fit_model_static(
    X_train_imp: np.ndarray,
    y_train: np.ndarray,
    model_args: dict,
    sample_weight: "np.ndarray | None" = None,
) -> dict:
    """Module-level model fitter; picklable for ProcessPoolExecutor."""
    if lgb is None:
        raise RuntimeError("lightgbm not installed; pip install lightgbm")
    use_es = model_args["dir_lgbm_early_stopping_rounds"] > 0
    n = len(y_train)
    if use_es:
        n_valid = max(
            int(n * float(model_args["dir_lgbm_valid_frac"])),
            int(model_args["dir_lgbm_valid_min"]),
        )
        n_valid = min(n_valid, n - 50)
        if n_valid < 1:
            use_es = False
    if use_es:
        split = n - n_valid
        X_tr, y_tr = X_train_imp[:split], y_train[:split]
        X_va, y_va = X_train_imp[split:], y_train[split:]
    else:
        X_tr, y_tr = X_train_imp, y_train
        X_va = y_va = None
    # When running parallel folds, override n_jobs=1 to avoid
    # thread oversubscription (wf_workers processes × lgbm threads).
    n_jobs = model_args["dir_lgbm_n_jobs"]
    if model_args.get("wf_workers", 1) > 1:
        n_jobs = 1
    clf = lgb.LGBMRegressor(
        boosting_type=model_args["dir_lgbm_boosting"],
        objective=model_args["dir_lgbm_objective"],
        learning_rate=model_args["dir_lgbm_learning_rate"],
        n_estimators=model_args["dir_lgbm_n_estimators"],
        num_leaves=model_args["dir_lgbm_num_leaves"],
        max_depth=model_args["dir_lgbm_max_depth"],
        min_child_samples=model_args["dir_lgbm_min_data_in_leaf"],
        min_sum_hessian_in_leaf=model_args["dir_lgbm_min_sum_hessian_in_leaf"],
        colsample_bytree=model_args["dir_lgbm_feature_fraction"],
        subsample=model_args["dir_lgbm_bagging_fraction"],
        subsample_freq=model_args["dir_lgbm_bagging_freq"],
        reg_alpha=model_args["dir_lgbm_lambda_l1"],
        reg_lambda=model_args["dir_lgbm_lambda_l2"],
        min_split_gain=model_args["dir_lgbm_min_gain_to_split"],
        max_bin=model_args["dir_lgbm_max_bin"],
        drop_rate=model_args["dir_lgbm_drop_rate"],
        skip_drop=model_args["dir_lgbm_skip_drop"],
        max_drop=model_args["dir_lgbm_max_drop"],
        uniform_drop=bool(model_args["dir_lgbm_uniform_drop"]),
        random_state=model_args["seed"],
        n_jobs=n_jobs,
        verbose=model_args["dir_lgbm_verbose"],
    )
    if use_es:
        # Split sample_weight to match the train/val split
        if sample_weight is not None:
            sw_tr = sample_weight[:split]
            sw_va = sample_weight[split:]
        else:
            sw_tr = sw_va = None
        clf.fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[sw_va],
            callbacks=[
                lgb.early_stopping(
                    model_args["dir_lgbm_early_stopping_rounds"], verbose=False
                )
            ],
        )
    else:
        clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    info = {"best_iter": getattr(clf, "best_iteration_", None)}
    return {"kind": "lgbm", "model": clf, "info": info}


def _predict_model_static(model_pack: dict, X_oos_imp: np.ndarray) -> np.ndarray:
    """Module-level model predictor; picklable for ProcessPoolExecutor."""
    return model_pack["model"].predict(X_oos_imp)


def _run_fold_worker(task: dict) -> dict:
    """
    Module-level fold worker: impute → fit → predict → metrics.
    Receives pre-extracted numpy arrays; never touches df_feat.
    Returns a result dict with keys: fold_idx, kind, metrics, oos_data, summary.
    """
    fold_idx = task["fold_idx"]
    X_train_raw = task["X_train_raw"]
    X_oos_raw = task["X_oos_raw"]
    y_tr = task["y_tr"]
    y_os = task["y_os"]
    model_args = task["model_args"]

    X_train_raw, X_oos_raw, all_nan_cols = _fill_all_nan_cols(
        X_train_raw, X_oos_raw
    )
    nan_frac = _nan_fraction_rows(X_train_raw)
    if nan_frac > MAX_TRAIN_NAN_FRAC:
        raise RuntimeError(
            f"fold {fold_idx}: too many NaNs in train rows: {nan_frac:.3f}"
        )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_oos_imp = imputer.transform(X_oos_raw)

    # ── Sample weight decay ───────────────────────────────────────────────
    # Positional exponential decay: newest sample weight = 1.0,
    # weight[i] = decay^(n-1-i) where i=0 is oldest.
    # Weights are then rescaled so sum == n (preserves effective sample size
    # interpretation in LightGBM's objective).
    # Only computed when decay > 0.
    sample_decay = float(model_args.get("sample_decay", 0.0))
    sample_weight: np.ndarray | None = None
    if sample_decay > 0.0:
        n_tr = len(y_tr)
        w = np.array(
            [sample_decay ** (n_tr - 1 - i) for i in range(n_tr)],
            dtype=float,
        )
        w_sum = w.sum()
        if w_sum > 0:
            sample_weight = w * (n_tr / w_sum)

    model_pack = _fit_model_static(X_train_imp, y_tr, model_args, sample_weight=sample_weight)
    y_pred = _predict_model_static(model_pack, X_oos_imp)
    # ── Optional: return train predictions for threshold calibration ──
    train_pred = None
    if task.get("return_train_pred", False):
        train_pred = _predict_model_static(model_pack, X_train_imp)
    model_best_iter = model_pack["info"].get("best_iter")

    if not np.isfinite(y_pred).all():
        raise RuntimeError(
            f"fold {fold_idx}: y_pred contains NaN/inf on OOS"
        )

    fold_dates = task["fold_dates"]
    metrics = {
        "fold_idx": fold_idx,
        "train_start": fold_dates["train_start"],
        "train_end": fold_dates["train_end"],
        "oos_start": fold_dates["oos_start"],
        "oos_end": fold_dates["oos_end"],
        "n_train": len(y_tr),
        "n_oos": len(y_os),
        "target_spearman": _spearman(y_pred, y_os),
        "target_rmse": _rmse(y_pred, y_os),
        "target_sign_acc": _sign_acc(y_pred, y_os),
        "skip_reason": "",
        "model": "lgbm",
        "model_best_iter": model_best_iter,
    }
    for pct in (0.10, 0.20, 0.30):
        stats = _top_k_stats(y_pred, y_os, pct)
        key = f"top{int(pct * 100)}"
        metrics[f"{key}_n_selected"] = stats["n_selected"]
        metrics[f"{key}_cov"] = stats["cov"]
        metrics[f"{key}_sign_acc"] = stats["sign_acc"]
        metrics[f"{key}_mean_pnl_proxy"] = stats["mean_pnl_proxy"]
        metrics[f"{key}_std_pnl_proxy"] = stats["std_pnl_proxy"]
        metrics[f"{key}_sharpe_proxy"] = stats["sharpe_proxy"]

    oos_data = {
        "ts_ms": task["ts_ms_oos"],
        "y_target": y_os,
        "target_pred": y_pred,
        "fold_idx": fold_idx,
        "context": task["context_data"],
    }

    summary = {
        "fold_idx": fold_idx,
        "all_nan_cols": all_nan_cols,
        "n_train_before_purge": task["n_train_before_purge"],
        "n_train_after_purge": task["n_train_after_purge"],
        "n_oos": len(y_os),
        "purge_bars": task["purge_bars"],
        "oos_start_pos": task["oos_start_pos"],
        "tr_start_idx": task["tr_start_idx"],
        "tr_end_idx": task["tr_end_idx"],
        "os_start_idx": task["os_start_idx"],
        "os_end_idx": task["os_end_idx"],
        "tr_ts_min": task["tr_ts_min"],
        "tr_ts_max": task["tr_ts_max"],
        "os_ts_min": task["os_ts_min"],
        "os_ts_max": task["os_ts_max"],
    }

    return {
        "fold_idx": fold_idx,
        "kind": "result",
        "metrics": metrics,
        "oos_data": oos_data,
        "summary": summary,
        "train_pred": train_pred,
    }


def _train_gate_lofo(
    preds_df: pd.DataFrame,
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    folds: list,
    args: argparse.Namespace,
    status: StatusBar,
) -> pd.DataFrame:
    """
    Train gate model via leave-one-fold-out cross-validation.

    For fold k:
      - Gate trains on all folds except k (sign(entry_pred) ? y_event as target)
      - Gate predicts on fold k OOS data ? gate_score

    Gate features = entry model features (abs_pred ??)
    Gate target = sign(entry_pred) ? y_event (?? ?? ? +|actual|, ??? ? -|actual|)

    Returns preds_df with 'gate_score' column added.
    """
    if lgb is None:
        raise RuntimeError("lightgbm not installed; required for gate model")

    gate_data = preds_df[["ts_ms", "target_pred", "y_target", "fold_idx"]].copy()
    gate_data["y_gate"] = np.sign(gate_data["target_pred"]) * gate_data["y_target"]

    feat_subset = df_feat[["ts_ms"] + feature_cols].copy()
    gate_data = gate_data.merge(feat_subset, on="ts_ms", how="left", sort=False)
    gate_feature_cols = list(feature_cols)

    gate_scores = np.full(len(gate_data), np.nan, dtype=float)
    unique_folds = sorted(gate_data["fold_idx"].dropna().unique().tolist())

    gate_fold_metrics: list[dict] = []

    for k in unique_folds:
        train_mask = gate_data["fold_idx"] != k
        test_mask = gate_data["fold_idx"] == k

        X_train = gate_data.loc[train_mask, gate_feature_cols].to_numpy(dtype=float)
        y_train = gate_data.loc[train_mask, "y_gate"].to_numpy(dtype=float)
        X_test = gate_data.loc[test_mask, gate_feature_cols].to_numpy(dtype=float)

        finite_mask = np.isfinite(y_train)
        if finite_mask.sum() < 50 or X_test.size == 0:
            continue
        X_train = X_train[finite_mask]
        y_train = y_train[finite_mask]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        gate_model = lgb.LGBMRegressor(
            objective="huber",
            boosting_type="gbdt",
            n_estimators=int(args.gate_lgbm_n_estimators),
            num_leaves=int(args.gate_lgbm_num_leaves),
            max_depth=int(args.gate_lgbm_max_depth),
            learning_rate=float(args.gate_lgbm_learning_rate),
            min_child_samples=int(args.gate_lgbm_min_data_in_leaf),
            colsample_bytree=float(args.gate_lgbm_feature_fraction),
            subsample=float(args.gate_lgbm_bagging_fraction),
            subsample_freq=int(args.gate_lgbm_bagging_freq),
            reg_lambda=float(args.gate_lgbm_lambda_l2),
            random_state=int(args.seed),
            n_jobs=1,
            verbose=-1,
        )

        n_tr = len(X_train_imp)
        n_val = max(int(n_tr * float(args.gate_lgbm_valid_frac)), 50)
        n_val = min(n_val, n_tr - 50)

        if n_val >= 50 and n_tr - n_val >= 1:
            split = n_tr - n_val
            gate_model.fit(
                X_train_imp[:split],
                y_train[:split],
                eval_set=[(X_train_imp[split:], y_train[split:])],
                callbacks=[
                    lgb.early_stopping(
                        int(args.gate_lgbm_early_stopping_rounds),
                        verbose=False,
                    )
                ],
            )
        else:
            gate_model.fit(X_train_imp, y_train)

        gate_pred = gate_model.predict(X_test_imp)
        gate_scores[test_mask.values] = gate_pred

        y_test = gate_data.loc[test_mask, "y_gate"].to_numpy(dtype=float)
        sp = _spearman(gate_pred, y_test)
        best_iter = getattr(gate_model, "best_iteration_", None)
        gate_fold_metrics.append(
            {
                "fold_idx": int(k),
                "n_train": int(train_mask.sum()),
                "gate_spearman": sp,
                "best_iter": best_iter,
            }
        )
        status.line(
            f"[GATE] fold {k}: n_train={int(train_mask.sum())} gate_sp={sp:.4f} best_iter={best_iter}"
        )

    preds_df = preds_df.copy()
    preds_df["gate_score"] = gate_scores
    preds_df.attrs["gate_fold_metrics"] = gate_fold_metrics
    return preds_df


def _download_zip_paths(
    symbol: str,
    start_ms: int,
    end_ms: int,
    cache_dir: str,
    dl_workers: int,
    rest_timeout: float,
    rest_sleep: float,
    rest_max_retries: int,
    status: StatusBar,
) -> list[str]:
    """Download/verify daily aggTrades zips; return ordered list of zip paths."""
    start_date = _dt.datetime.utcfromtimestamp(start_ms / 1000).date()
    end_date = _dt.datetime.utcfromtimestamp((end_ms - 1) / 1000).date()
    dates_to_process = []
    d = start_date
    while d <= end_date:
        dates_to_process.append(d)
        d += _dt.timedelta(days=1)

    path_by_date: dict[_dt.date, str] = {}
    missing_dates = []
    total = len(dates_to_process)
    for i, day in enumerate(dates_to_process, start=1):
        dstr = day.strftime("%Y-%m-%d")
        fname = f"{symbol}-aggTrades-{dstr}.zip"
        rel = os.path.join("aggTrades", symbol, fname)
        fpath = os.path.join(cache_dir, rel)
        checksum_path = fpath + ".CHECKSUM"
        status.update("DL", f"verify {i}/{total} {dstr}")
        cached = False
        if os.path.exists(fpath) and os.path.exists(checksum_path):
            try:
                with open(checksum_path, "r", encoding="utf-8") as f:
                    algo, chk = _parse_checksum_text(f.read())
                if _hash_file(fpath, algo) == chk:
                    path_by_date[day] = fpath
                    cached = True
            except Exception:
                pass
        if not cached:
            missing_dates.append(day)

    if missing_dates:
        status.update(
            "DL",
            f"downloading {len(missing_dates)}/{total} files "
            f"workers={dl_workers}",
            force=True,
        )
        with ThreadPoolExecutor(max_workers=int(dl_workers)) as ex:
            futures = {
                ex.submit(
                    download_daily_aggtrades,
                    symbol,
                    day,
                    cache_dir,
                    float(rest_timeout),
                    float(rest_sleep),
                    int(rest_max_retries),
                ): day
                for day in missing_dates
            }
            done = 0
            for fut in as_completed(futures):
                day = futures[fut]
                path = fut.result()
                path_by_date[day] = path
                done += 1
                status.update("DL", f"done {done}/{len(missing_dates)}", force=True)

    all_paths = [path_by_date[d] for d in dates_to_process]
    status.done("DL", f"ready days={len(all_paths)}")
    return all_paths


def _build_features(
    all_paths: list[str],
    start_ms: int,
    end_ms: int,
    v_target: float,
    feat_scale: int,
    n_lags: int,
    pctrank_window: int,
    max_bars: int,
    symbol: str,
    status: StatusBar,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build VBs → VPIN → features. Returns (df_vb, df_feat, feature_cols)."""
    status.update("VB", "building volume bars (chunked)...", force=True)
    df_vb = build_volume_bars_chunked(
        all_paths,
        start_ms,
        end_ms,
        float(v_target),
    )
    status.done("VB", f"bars={len(df_vb):,}")
    if df_vb.empty:
        raise RuntimeError("no volume bars built")
    ensure_monotonic(df_vb["ts_ms"].to_numpy(), "volume bars ts_ms")

    if max_bars > 0 and len(df_vb) > max_bars:
        df_vb = df_vb.iloc[: int(max_bars)].copy().reset_index(drop=True)
        status.line(f"[VB] trimmed to max_bars={max_bars}")

    N_vpin = max(10, int(feat_scale))
    imb = df_vb["taker_buy_vol"] - df_vb["taker_sell_vol"]
    vpin_num = imb.abs().rolling(N_vpin, min_periods=max(3, N_vpin // 4)).sum()
    vpin_den = df_vb["volume"].rolling(N_vpin, min_periods=max(3, N_vpin // 4)).sum()
    df_vb["vpin_N"] = safe_div(vpin_num, vpin_den)
    df_vb["vpin_delta"] = df_vb["vpin_N"].diff(1)

    df_feat = _make_feature_df(
        df_vb,
        int(feat_scale),
        int(n_lags),
        int(pctrank_window),
    )

    feature_cols = _build_feature_cols(int(n_lags))
    df_feat = df_feat.sort_values("ts_ms").reset_index(drop=True)
    ts_vals = pd.to_numeric(df_feat["ts_ms"], errors="coerce")
    if not np.isfinite(ts_vals.to_numpy()).all():
        print("[warn] ts_ms has non-finite values; dropping those rows")
        df_feat = df_feat[np.isfinite(ts_vals.to_numpy())].copy()
        ts_vals = pd.to_numeric(df_feat["ts_ms"], errors="coerce")
    df_feat["ts_ms"] = ts_vals.astype("int64")
    df_feat = df_feat.sort_values("ts_ms").reset_index(drop=True)
    df_feat["date"] = pd.to_datetime(df_feat["ts_ms"], unit="ms", utc=True).dt.date
    df_feat["symbol"] = symbol

    nan_any = float(np.mean(np.isnan(df_feat[feature_cols].to_numpy()).any(axis=1)))
    print(f"[inv] n_bars={len(df_feat)} nan_any_frac={nan_any:.3f}")

    return df_vb, df_feat, feature_cols


def _save_results(
    outdir: str,
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    summary_rows: list[dict],
    gate_enable: bool,
    gate_metrics: dict | None,
    args: argparse.Namespace,
    feature_cols: list[str],
    df_feat: pd.DataFrame,
    target_col: str,
) -> None:
    """Save metrics.csv, preds.csv, summary.json, gate_metrics.json, run.json."""
    metrics_path = os.path.join(outdir, "metrics.csv")
    preds_path = os.path.join(outdir, "preds.csv")
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    summary = {
        "metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "metrics_median": metrics_df.median(numeric_only=True).to_dict(),
        "folds": summary_rows,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if gate_enable and gate_metrics is not None:
        with open(os.path.join(outdir, "gate_metrics.json"), "w") as f:
            json.dump(gate_metrics, f, indent=2)

    run_info = {
        "args": vars(args),
        "feature_cols": feature_cols,
        "data_stats": {
            "n_bars": int(len(df_feat)),
            "start_ts": int(df_feat["ts_ms"].min()),
            "end_ts": int(df_feat["ts_ms"].max()),
            "v_target": float(args.v_target),
            "H": int(args.horizon),
            "feat_scale": int(args.feat_scale),
            "vol_ewma_span": int(args.vol_ewma_span),
            "target": target_col,
            "target_mode": args.target_mode,
            "n_lags": int(args.n_lags),
            "n_features": len(feature_cols),
            "wf_workers": int(args.wf_workers),
            "wf_expanding": bool(args.wf_expanding),
            "wf_sample_decay": float(args.wf_sample_decay),
        },
        "artifacts": {
            "metrics": metrics_path,
            "preds": preds_path,
            "summary": os.path.join(outdir, "summary.json"),
        },
    }
    with open(os.path.join(outdir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)


def _run_continuous_alpha(
    args: argparse.Namespace,
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    status: StatusBar,
) -> None:
    """
    Continuous Alpha pipeline: Alpha Model → Entry Model → Fixed-H exit.
    """
    ca_H = int(args.ca_horizon)
    top_pct = float(args.ca_alpha_top_pct)
    skip_alpha_oos = bool(getattr(args, "ca_skip_alpha_oos", 0))
    vol_span = int(args.vol_ewma_span)

    status.update("CA", "computing alpha target...", force=True)
    df_feat["alpha_target"] = compute_alpha_target(df_feat, ca_H, vol_span)
    n_valid_alpha = int(df_feat["alpha_target"].notna().sum())
    status.line(f"[CA] alpha_target valid={n_valid_alpha:,}")

    status.update("CA", "computing entry target...", force=True)
    df_feat["y_entry"] = compute_vwap_entry_target(df_feat, ca_H, vol_span)
    n_valid_entry = int(df_feat["y_entry"].notna().sum())
    status.line(f"[CA] y_entry valid={n_valid_entry:,}")

    folds = _build_wf_folds_bars(
        len(df_feat),
        int(args.wf_train_bars),
        int(args.wf_oos_bars),
        int(args.wf_step_bars),
        int(args.wf_gap_bars),
    )
    if not folds:
        raise RuntimeError("no folds constructed for continuous_alpha")

    purge_bars = int(args.wf_purge_bars)
    if purge_bars < 0:
        purge_bars = ca_H
    else:
        purge_bars = max(purge_bars, ca_H)

    model_args = {
        "seed": int(args.seed),
        "dir_lgbm_boosting": args.dir_lgbm_boosting,
        "dir_lgbm_objective": args.dir_lgbm_objective,
        "dir_lgbm_learning_rate": float(args.dir_lgbm_learning_rate),
        "dir_lgbm_n_estimators": int(args.dir_lgbm_n_estimators),
        "dir_lgbm_num_leaves": int(args.dir_lgbm_num_leaves),
        "dir_lgbm_max_depth": int(args.dir_lgbm_max_depth),
        "dir_lgbm_min_data_in_leaf": int(args.dir_lgbm_min_data_in_leaf),
        "dir_lgbm_min_sum_hessian_in_leaf": float(args.dir_lgbm_min_sum_hessian_in_leaf),
        "dir_lgbm_feature_fraction": float(args.dir_lgbm_feature_fraction),
        "dir_lgbm_bagging_fraction": float(args.dir_lgbm_bagging_fraction),
        "dir_lgbm_bagging_freq": int(args.dir_lgbm_bagging_freq),
        "dir_lgbm_lambda_l1": float(args.dir_lgbm_lambda_l1),
        "dir_lgbm_lambda_l2": float(args.dir_lgbm_lambda_l2),
        "dir_lgbm_min_gain_to_split": float(args.dir_lgbm_min_gain_to_split),
        "dir_lgbm_max_bin": int(args.dir_lgbm_max_bin),
        "dir_lgbm_drop_rate": float(args.dir_lgbm_drop_rate),
        "dir_lgbm_skip_drop": float(args.dir_lgbm_skip_drop),
        "dir_lgbm_max_drop": int(args.dir_lgbm_max_drop),
        "dir_lgbm_uniform_drop": bool(args.dir_lgbm_uniform_drop),
        "dir_lgbm_n_jobs": int(args.dir_lgbm_n_jobs),
        "dir_lgbm_verbose": int(args.dir_lgbm_verbose),
        "dir_lgbm_early_stopping_rounds": int(args.dir_lgbm_early_stopping_rounds),
        "dir_lgbm_valid_frac": float(args.dir_lgbm_valid_frac),
        "dir_lgbm_valid_min": int(args.dir_lgbm_valid_min),
        "wf_workers": int(args.wf_workers),
        "sample_decay": float(args.wf_sample_decay),
    }

    idx = np.arange(len(df_feat))
    alpha_target_arr = df_feat["alpha_target"].to_numpy(dtype=float)
    entry_target_arr = df_feat["y_entry"].to_numpy(dtype=float)

    alpha_metrics_rows: list[dict] = []
    entry_metrics_rows: list[dict] = []
    preds_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    status.line(
        f"[CA] folds={len(folds)} purge={purge_bars} ca_H={ca_H} top_pct={top_pct}"
    )

    for fold_idx, fold in enumerate(folds):
        tr_start_idx, tr_end_idx, gap_end_idx, os_end_idx = fold

        if args.wf_expanding:
            mask_tr = idx < tr_end_idx
        else:
            mask_tr = (idx >= tr_start_idx) & (idx < tr_end_idx)
        mask_os = (idx >= gap_end_idx) & (idx < os_end_idx)

        oos_start_pos = gap_end_idx
        keep_train = idx < (oos_start_pos - purge_bars)

        mask_tr_alpha = mask_tr & np.isfinite(alpha_target_arr) & keep_train
        mask_os_alpha = mask_os & np.isfinite(alpha_target_arr)

        n_tr_alpha = int(mask_tr_alpha.sum())
        n_os_alpha = int(mask_os_alpha.sum())

        if n_tr_alpha < 100 or n_os_alpha < 50:
            summary_rows.append(
                {
                    "fold_idx": fold_idx,
                    "skip_reason": "insufficient_alpha_samples",
                    "n_tr_alpha": n_tr_alpha,
                    "n_os_alpha": n_os_alpha,
                }
            )
            status.line(f"[CA] fold {fold_idx}: skip (tr={n_tr_alpha}, os={n_os_alpha})")
            continue

        X_tr_alpha = df_feat.loc[mask_tr_alpha, feature_cols].to_numpy(dtype=float)
        X_os_alpha = df_feat.loc[mask_os_alpha, feature_cols].to_numpy(dtype=float)
        y_tr_alpha = alpha_target_arr[mask_tr_alpha]
        y_os_alpha = alpha_target_arr[mask_os_alpha]

        alpha_task = {
            "fold_idx": fold_idx,
            "X_train_raw": X_tr_alpha,
            "X_oos_raw": X_os_alpha,
            "y_tr": y_tr_alpha,
            "y_os": y_os_alpha,
            "ts_ms_oos": df_feat.loc[mask_os_alpha, "ts_ms"].to_numpy().copy(),
            "ts_ms_tr": df_feat.loc[mask_tr_alpha, "ts_ms"].to_numpy().copy(),
            "context_data": {},
            "fold_dates": {
                "train_start": int(df_feat.loc[mask_tr_alpha, "ts_ms"].min()),
                "train_end": int(df_feat.loc[mask_tr_alpha, "ts_ms"].max()),
                "oos_start": int(df_feat.loc[mask_os_alpha, "ts_ms"].min()),
                "oos_end": int(df_feat.loc[mask_os_alpha, "ts_ms"].max()),
            },
            "n_train_before_purge": int((mask_tr & np.isfinite(alpha_target_arr)).sum()),
            "n_train_after_purge": n_tr_alpha,
            "oos_start_pos": oos_start_pos,
            "tr_start_idx": int(np.where(mask_tr_alpha)[0].min()),
            "tr_end_idx": int(np.where(mask_tr_alpha)[0].max()) + 1,
            "os_start_idx": int(np.where(mask_os_alpha)[0].min()),
            "os_end_idx": int(np.where(mask_os_alpha)[0].max()) + 1,
            "tr_ts_min": int(df_feat.loc[mask_tr_alpha, "ts_ms"].min()),
            "tr_ts_max": int(df_feat.loc[mask_tr_alpha, "ts_ms"].max()),
            "os_ts_min": int(df_feat.loc[mask_os_alpha, "ts_ms"].min()),
            "os_ts_max": int(df_feat.loc[mask_os_alpha, "ts_ms"].max()),
            "purge_bars": purge_bars,
            "model_args": model_args,
            "return_train_pred": True,
        }
        alpha_result = _run_fold_worker(alpha_task)
        alpha_pred_oos = alpha_result["oos_data"]["target_pred"]
        alpha_pred_train = alpha_result["train_pred"]
        alpha_metrics_rows.append(alpha_result["metrics"])

        if not skip_alpha_oos:
            # ── Rolling percentile threshold (no look-ahead) ────────────
            pct_cutoff = (1.0 - top_pct) * 100
            score_pool = list(alpha_pred_train)
            alpha_pass_oos = np.zeros(len(alpha_pred_oos), dtype=bool)
            alpha_thresholds_rolling = np.zeros(len(alpha_pred_oos), dtype=float)

            for i in range(len(alpha_pred_oos)):
                thr = float(np.percentile(score_pool, pct_cutoff))
                alpha_thresholds_rolling[i] = thr
                if alpha_pred_oos[i] >= thr:
                    alpha_pass_oos[i] = True
                score_pool.append(alpha_pred_oos[i])

            alpha_threshold_oos = float(np.mean(alpha_thresholds_rolling))
            n_alpha_pass = int(alpha_pass_oos.sum())

            if n_alpha_pass < 20:
                summary_rows.append(
                    {
                        "fold_idx": fold_idx,
                        "skip_reason": "too_few_alpha_pass",
                        "n_alpha_pass": n_alpha_pass,
                        "alpha_threshold": alpha_threshold_oos,
                    }
                )
                status.line(f"[CA] fold {fold_idx}: alpha pass too few ({n_alpha_pass})")
                continue

        # ── Entry training (unchanged for both paths) ────────────
        mask_tr_entry_base = mask_tr & np.isfinite(entry_target_arr) & keep_train
        mask_tr_entry_base = mask_tr_entry_base & np.isfinite(alpha_target_arr)
        alpha_vals_for_entry_tr = alpha_target_arr[mask_tr_entry_base]

        if len(alpha_vals_for_entry_tr) < 100:
            summary_rows.append(
                {
                    "fold_idx": fold_idx,
                    "skip_reason": "insufficient_entry_train",
                    "n_available": len(alpha_vals_for_entry_tr),
                }
            )
            continue

        alpha_threshold_train = float(
            np.nanpercentile(alpha_vals_for_entry_tr, (1.0 - top_pct) * 100)
        )
        mask_tr_entry = mask_tr_entry_base & (alpha_target_arr >= alpha_threshold_train)
        n_tr_entry = int(mask_tr_entry.sum())

        if n_tr_entry < 50:
            summary_rows.append(
                {
                    "fold_idx": fold_idx,
                    "skip_reason": "too_few_entry_train",
                    "n_tr_entry": n_tr_entry,
                }
            )
            continue

        # ── OOS bar selection ────────────────────────────────────
        os_alpha_indices = np.where(mask_os_alpha)[0]

        if skip_alpha_oos:
            os_entry_valid = np.isfinite(entry_target_arr[os_alpha_indices])
            os_entry_indices = os_alpha_indices[os_entry_valid]
            n_os_entry = len(os_entry_indices)

            alpha_threshold_oos = float("nan")
            n_alpha_pass = n_os_entry

            alpha_feat_os = alpha_pred_oos[os_entry_valid].reshape(-1, 1)

            alpha_scores_for_output = alpha_pred_oos[os_entry_valid]
            alpha_thresholds_for_output = np.full(n_os_entry, float("nan"))
        else:
            os_entry_indices = os_alpha_indices[alpha_pass_oos]
            os_entry_valid = np.isfinite(entry_target_arr[os_entry_indices])
            os_entry_indices = os_entry_indices[os_entry_valid]
            n_os_entry = len(os_entry_indices)

            alpha_pass_positions = np.where(alpha_pass_oos)[0]
            alpha_pass_and_valid = alpha_pass_positions[os_entry_valid]
            alpha_feat_os = alpha_pred_oos[alpha_pass_and_valid].reshape(-1, 1)

            alpha_scores_for_output = alpha_pred_oos[alpha_pass_and_valid]
            alpha_thresholds_for_output = alpha_thresholds_rolling[alpha_pass_positions][
                os_entry_valid
            ]

        if n_os_entry < 10:
            summary_rows.append(
                {
                    "fold_idx": fold_idx,
                    "skip_reason": "too_few_entry_oos",
                    "n_os_entry": n_os_entry,
                }
            )
            continue

        # ── Entry features (common for both paths) ───────────────
        entry_feature_cols = feature_cols + ["_alpha_feat"]

        X_tr_entry_base = df_feat.loc[mask_tr_entry, feature_cols].to_numpy(dtype=float)
        alpha_feat_tr = alpha_target_arr[mask_tr_entry].reshape(-1, 1)
        X_tr_entry = np.hstack([X_tr_entry_base, alpha_feat_tr])
        y_tr_entry = entry_target_arr[mask_tr_entry]

        X_os_entry_base = df_feat.iloc[os_entry_indices][feature_cols].to_numpy(dtype=float)
        X_os_entry = np.hstack([X_os_entry_base, alpha_feat_os])
        y_os_entry = entry_target_arr[os_entry_indices]

        entry_task = {
            "fold_idx": fold_idx,
            "X_train_raw": X_tr_entry,
            "X_oos_raw": X_os_entry,
            "y_tr": y_tr_entry,
            "y_os": y_os_entry,
            "ts_ms_oos": df_feat.iloc[os_entry_indices]["ts_ms"].to_numpy().copy(),
            "ts_ms_tr": df_feat.loc[mask_tr_entry, "ts_ms"].to_numpy().copy(),
            "context_data": {},
            "fold_dates": {
                "train_start": int(df_feat.loc[mask_tr_entry, "ts_ms"].min()),
                "train_end": int(df_feat.loc[mask_tr_entry, "ts_ms"].max()),
                "oos_start": int(df_feat.iloc[os_entry_indices]["ts_ms"].min()),
                "oos_end": int(df_feat.iloc[os_entry_indices]["ts_ms"].max()),
            },
            "n_train_before_purge": n_tr_entry,
            "n_train_after_purge": n_tr_entry,
            "oos_start_pos": oos_start_pos,
            "tr_start_idx": int(np.where(mask_tr_entry)[0].min()),
            "tr_end_idx": int(np.where(mask_tr_entry)[0].max()) + 1,
            "os_start_idx": int(os_entry_indices.min()),
            "os_end_idx": int(os_entry_indices.max()) + 1,
            "tr_ts_min": int(df_feat.loc[mask_tr_entry, "ts_ms"].min()),
            "tr_ts_max": int(df_feat.loc[mask_tr_entry, "ts_ms"].max()),
            "os_ts_min": int(df_feat.iloc[os_entry_indices]["ts_ms"].min()),
            "os_ts_max": int(df_feat.iloc[os_entry_indices]["ts_ms"].max()),
            "purge_bars": purge_bars,
            "model_args": model_args,
        }
        entry_result = _run_fold_worker(entry_task)
        entry_pred_oos = entry_result["oos_data"]["target_pred"]
        entry_metrics_rows.append(entry_result["metrics"])

        oos_rows = pd.DataFrame(
            {
                "ts_ms": df_feat.iloc[os_entry_indices]["ts_ms"].values,
                "fold_idx": fold_idx,
                "alpha_score": alpha_scores_for_output,
                "alpha_target_actual": alpha_target_arr[os_entry_indices],
                "y_entry_target": y_os_entry,
                "entry_pred": entry_pred_oos,
                "alpha_threshold_oos": alpha_thresholds_for_output,
                "alpha_threshold_train": alpha_threshold_train,
            }
        )
        preds_rows.append(oos_rows)

        summary_rows.append(
            {
                "fold_idx": fold_idx,
                "n_tr_alpha": n_tr_alpha,
                "n_os_alpha": n_os_alpha,
                "n_alpha_pass": n_alpha_pass,
                "alpha_threshold_oos": alpha_threshold_oos,
                "alpha_threshold_train": alpha_threshold_train,
                "n_tr_entry": n_tr_entry,
                "n_os_entry": n_os_entry,
                "alpha_spearman": alpha_result["metrics"]["target_spearman"],
                "entry_spearman": entry_result["metrics"]["target_spearman"],
                "entry_sign_acc": entry_result["metrics"]["target_sign_acc"],
            }
        )

        status.line(
            f"[CA] fold {fold_idx}: "
            f"alpha_sp={alpha_result['metrics']['target_spearman']:.4f} "
            f"pass={n_alpha_pass}/{n_os_alpha} "
            f"entry_sp={entry_result['metrics']['target_spearman']:.4f} "
            f"entry_sign={entry_result['metrics']['target_sign_acc']:.3f}"
        )

    if not entry_metrics_rows:
        raise RuntimeError("continuous_alpha: no folds produced entry metrics")

    alpha_metrics_df = pd.DataFrame(alpha_metrics_rows)
    entry_metrics_df = pd.DataFrame(entry_metrics_rows)

    preds_df = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()

    if not preds_df.empty:
        y_pred = preds_df["entry_pred"].to_numpy(dtype=float)
        y_true = preds_df["y_entry_target"].to_numpy(dtype=float)
        mask_valid = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred_v = y_pred[mask_valid]
        y_true_v = y_true[mask_valid]

        overall_spearman = _spearman(y_pred_v, y_true_v)
        overall_sign_acc = _sign_acc(y_pred_v, y_true_v)

        pnl_proxy = np.sign(y_pred_v) * y_true_v
        overall_mean_pnl = float(np.mean(pnl_proxy))
        overall_sharpe = float(np.mean(pnl_proxy) / (np.std(pnl_proxy) + EPS))

        top_stats = {}
        for pct in (0.10, 0.20, 0.30):
            stats = _top_k_stats(y_pred_v, y_true_v, pct)
            top_stats[f"top{int(pct * 100)}"] = stats
    else:
        overall_spearman = float("nan")
        overall_sign_acc = float("nan")
        overall_mean_pnl = float("nan")
        overall_sharpe = float("nan")
        top_stats = {}

    summary = {
        "mode": "continuous_alpha",
        "ca_horizon": ca_H,
        "ca_alpha_top_pct": top_pct,
        "ca_skip_alpha_oos": skip_alpha_oos,
        "feat_scale": int(args.feat_scale),
        "vol_ewma_span": vol_span,
        "n_folds": len(folds),
        "n_folds_completed": len(entry_metrics_rows),
        "alpha_model": {
            "metrics_mean": alpha_metrics_df.mean(numeric_only=True).to_dict()
            if not alpha_metrics_df.empty
            else {},
            "metrics_std": alpha_metrics_df.std(numeric_only=True).to_dict()
            if not alpha_metrics_df.empty
            else {},
        },
        "entry_model": {
            "metrics_mean": entry_metrics_df.mean(numeric_only=True).to_dict()
            if not entry_metrics_df.empty
            else {},
            "metrics_std": entry_metrics_df.std(numeric_only=True).to_dict()
            if not entry_metrics_df.empty
            else {},
        },
        "overall": {
            "spearman": overall_spearman,
            "sign_acc": overall_sign_acc,
            "mean_pnl_proxy": overall_mean_pnl,
            "sharpe_proxy": overall_sharpe,
            "n_trades": len(preds_df),
            "top_stats": top_stats,
        },
        "folds": summary_rows,
    }

    os.makedirs(args.outdir, exist_ok=True)
    alpha_metrics_df.to_csv(os.path.join(args.outdir, "alpha_metrics.csv"), index=False)
    entry_metrics_df.to_csv(os.path.join(args.outdir, "entry_metrics.csv"), index=False)
    if not preds_df.empty:
        preds_df.to_csv(os.path.join(args.outdir, "ca_preds.csv"), index=False)

    with open(os.path.join(args.outdir, "ca_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    run_info = {
        "args": vars(args),
        "feature_cols": feature_cols,
        "data_stats": {
            "n_bars": int(len(df_feat)),
            "start_ts": int(df_feat["ts_ms"].min()),
            "end_ts": int(df_feat["ts_ms"].max()),
            "v_target": float(args.v_target),
            "feat_scale": int(args.feat_scale),
            "ca_horizon": ca_H,
            "vol_ewma_span": vol_span,
            "target_mode": "continuous_alpha",
            "n_lags": int(args.n_lags),
            "n_features": len(feature_cols),
        },
        "artifacts": {
            "alpha_metrics": os.path.join(args.outdir, "alpha_metrics.csv"),
            "entry_metrics": os.path.join(args.outdir, "entry_metrics.csv"),
            "ca_preds": os.path.join(args.outdir, "ca_preds.csv"),
            "ca_summary": os.path.join(args.outdir, "ca_summary.json"),
        },
    }
    with open(os.path.join(args.outdir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, default=str)

    status.done(
        "CA",
        f"spearman={overall_spearman:.4f} sign_acc={overall_sign_acc:.3f} "
        f"sharpe={overall_sharpe:.3f} trades={len(preds_df)}",
    )


def _build_wf_folds_bars(
    n_rows: int,
    train_bars: int,
    oos_bars: int,
    step_bars: int,
    gap_bars: int,
) -> list[tuple[int, int, int, int]]:
    folds = []
    cursor = 0
    step = step_bars if step_bars > 0 else oos_bars
    while True:
        tr_start_idx = cursor
        tr_end_idx = cursor + train_bars
        gap_end_idx = tr_end_idx + gap_bars
        os_end_idx = gap_end_idx + oos_bars
        if os_end_idx > n_rows:
            break
        folds.append((tr_start_idx, tr_end_idx, gap_end_idx, os_end_idx))
        cursor += step
    return folds


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    start_ms = parse_date_to_utc_ms(args.start)
    end_ms = parse_date_to_utc_ms(args.end)
    if end_ms <= start_ms:
        raise ValueError("--end must be after --start")
    if args.v_target <= 0:
        raise ValueError("--v_target must be > 0")
    days = (end_ms - start_ms) / (24 * 3600 * 1000)
    if args.wf_mode == "days":
        min_days = args.wf_train_days + args.wf_oos_days + 1
        if days < min_days:
            print(f"date range too small for WF (wf_mode=days): have={days:.1f} need>={min_days}; exiting")
            return
    else:
        if (
            args.wf_train_days != 30
            or args.wf_oos_days != 7
            or args.wf_step_days != 7
            or args.wf_gap_days != 0
        ):
            print("[info] wf_mode=bars: wf_*_days flags are ignored (bars splitting uses wf_*_bars).")

    status = StatusBar(
        enabled=bool(args.status_bar),
        every_sec=float(args.status_every_sec),
        width=int(args.status_width),
    )

    all_paths = _download_zip_paths(
        symbol=args.symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        cache_dir=args.cache_dir,
        dl_workers=int(args.dl_workers),
        rest_timeout=float(args.rest_timeout),
        rest_sleep=float(args.rest_sleep),
        rest_max_retries=int(args.rest_max_retries),
        status=status,
    )

    df_vb, df_feat, feature_cols = _build_features(
        all_paths=all_paths,
        start_ms=start_ms,
        end_ms=end_ms,
        v_target=float(args.v_target),
        feat_scale=int(args.feat_scale),
        n_lags=int(args.n_lags),
        pctrank_window=int(args.pctrank_window),
        max_bars=int(args.max_bars),
        symbol=args.symbol,
        status=status,
    )

    H = int(args.horizon)
    target_col: str

    if args.target_mode == "vtsr":
        # ── VTSR target (original path, unchanged) ───────────────────────
        df_feat["y_vtsr"] = compute_vtsr_target(
            df_feat, H, int(args.vol_ewma_span), eps=EPS
        )
        if H > 0 and df_feat["y_vtsr"].tail(H).notna().any():
            raise RuntimeError(
                "last H rows of y_vtsr must be NaN (future leakage check failed)"
            )
        target_col = "y_vtsr"
        status.line(
            f"[TARGET] mode=vtsr H={H} "
            f"coverage={df_feat['y_vtsr'].notna().mean():.3f}"
        )

    elif args.target_mode == "rt3_event":
        # ── RT3 event target ────────────────────────────────────────────
        status.update("RT3", "computing trigger...", force=True)
        rt3 = compute_rt3_trigger(
            df_feat,
            spread_window=int(args.rt3_spread_window),
            spread_pct_on=float(args.rt3_spread_pct_on),
            spread_pct_off=float(args.rt3_spread_pct_off),
            K=int(args.rt3_k),
            imb_consistency_on=float(args.rt3_imb_consistency_on),
            imb_consistency_off=float(args.rt3_imb_consistency_off),
            off_drop_ratio=float(args.rt3_off_drop_ratio),
            off_peak_window=int(args.rt3_off_peak_window),
            off_confirm_bars=int(args.rt3_off_confirm_bars),
        )
        # Attach trigger columns to df_feat for inspection / context output
        df_feat["rt3_on"] = rt3["rt3_on"]
        df_feat["rt3_off"] = rt3["rt3_off"]
        df_feat["entry_direction"] = rt3["entry_direction"]
        df_feat["spread_pct"] = rt3["spread_pct"]
        df_feat["imb_consistency"] = rt3["imb_consistency"]

        n_on = int(rt3["rt3_on"].sum())
        status.done(
            "RT3",
            f"trigger ON bars={n_on:,} "
            f"({n_on / max(len(df_feat), 1):.3f} of total)"
        )
        if n_on < 100:
            raise RuntimeError(
                f"RT3 trigger fired only {n_on} times; "
                "too few for WF training. Relax --rt3_spread_pct_on "
                "or --rt3_spread_window."
            )

        status.update("TARGET", "computing event target...", force=True)
        df_feat["y_event"] = compute_event_target(
            df_feat,
            rt3=rt3,
            vol_ewma_span=int(args.vol_ewma_span),
            hmax=int(args.rt3_hmax),
            eps=EPS,
        )

        target_col = "y_event"
        n_event = int(df_feat["y_event"].notna().sum())
        coverage = n_event / max(n_on, 1)
        status.done(
            "TARGET",
            f"mode=rt3_event events={n_event:,} "
            f"coverage_of_trigger={coverage:.3f} "
            f"hmax={args.rt3_hmax}"
        )
        if n_event < 100:
            raise RuntimeError(
                f"y_event has only {n_event} valid labels; "
                "too few for WF training. Check RT3 parameters."
            )
        # Leakage check: every valid y_event[t] must reference only t+1..t+hmax
        # (guaranteed by compute_event_target loop structure — no further check needed)

    elif args.target_mode == "continuous_alpha":
        _run_continuous_alpha(args, df_feat, feature_cols, status)
        return

    # walk-forward splits
    if args.wf_mode == "bars":
        print("[info] wf_mode=bars: ignoring wf_*_days flags")
        need = args.wf_train_bars + args.wf_gap_bars + args.wf_oos_bars + 1
        if len(df_feat) < need:
            raise RuntimeError(f"not enough rows for bars WF: need>={need} got={len(df_feat)}")
        folds = _build_wf_folds_bars(
            len(df_feat),
            args.wf_train_bars,
            args.wf_oos_bars,
            args.wf_step_bars,
            args.wf_gap_bars,
        )
    else:
        if args.wf_train_bars or args.wf_oos_bars or args.wf_step_bars or args.wf_gap_bars:
            print("[info] wf_mode=days: ignoring wf_*_bars flags")
        date_series = df_feat["date"].dropna()
        dates = sorted(date_series.unique().tolist())
        if not dates:
            raise RuntimeError("no valid dates after feature build (date column empty)")
        folds = []
        i = 0
        while i < len(dates):
            train_start = dates[i]
            train_end = train_start + _dt.timedelta(days=args.wf_train_days)
            gap_end = train_end + _dt.timedelta(days=args.wf_gap_days)
            oos_end = gap_end + _dt.timedelta(days=args.wf_oos_days)
            if oos_end > dates[-1]:
                break
            folds.append((train_start, train_end, gap_end, oos_end))
            i += args.wf_step_days
    if not folds:
        raise RuntimeError("no folds constructed")

    gate_enable = int(args.gate_enable) == 1
    gate_em_rounds = int(args.gate_em_rounds)
    em_rounds_total = max(1, gate_em_rounds + 1) if gate_enable else 1
    gate_pass_mask: np.ndarray | None = None
    prev_sp_gated: float | None = None
    gate_metrics: dict | None = None
    em_history: list[dict] = []

    metrics_df: pd.DataFrame | None = None
    preds_df: pd.DataFrame | None = None
    summary_rows: list[dict] = []
    em_rounds_completed = 0

    for em_round in range(em_rounds_total):
        metrics_rows = []
        preds_rows = []
        summary_rows = []
        purge_bars = int(args.wf_purge_bars)
        # For event target, max label lookahead is hmax bars, not horizon.
        # purge_bars must cover the full label lookahead to prevent leakage.
        if args.target_mode == "rt3_event":
            label_lookahead = int(args.rt3_hmax)
        elif args.target_mode == "continuous_alpha":
            label_lookahead = int(args.ca_horizon)
        else:
            label_lookahead = int(args.horizon)
        if purge_bars < 0:
            purge_bars = label_lookahead
        else:
            purge_bars = max(purge_bars, label_lookahead)

        # Build model_args dict once; passed to every fold task.
        # Plain dict is picklable; avoids passing argparse.Namespace to workers.
        model_args = {
            "seed": int(args.seed),
            "dir_lgbm_boosting": args.dir_lgbm_boosting,
            "dir_lgbm_objective": args.dir_lgbm_objective,
            "dir_lgbm_learning_rate": float(args.dir_lgbm_learning_rate),
            "dir_lgbm_n_estimators": int(args.dir_lgbm_n_estimators),
            "dir_lgbm_num_leaves": int(args.dir_lgbm_num_leaves),
            "dir_lgbm_max_depth": int(args.dir_lgbm_max_depth),
            "dir_lgbm_min_data_in_leaf": int(args.dir_lgbm_min_data_in_leaf),
            "dir_lgbm_min_sum_hessian_in_leaf": float(args.dir_lgbm_min_sum_hessian_in_leaf),
            "dir_lgbm_feature_fraction": float(args.dir_lgbm_feature_fraction),
            "dir_lgbm_bagging_fraction": float(args.dir_lgbm_bagging_fraction),
            "dir_lgbm_bagging_freq": int(args.dir_lgbm_bagging_freq),
            "dir_lgbm_lambda_l1": float(args.dir_lgbm_lambda_l1),
            "dir_lgbm_lambda_l2": float(args.dir_lgbm_lambda_l2),
            "dir_lgbm_min_gain_to_split": float(args.dir_lgbm_min_gain_to_split),
            "dir_lgbm_max_bin": int(args.dir_lgbm_max_bin),
            "dir_lgbm_drop_rate": float(args.dir_lgbm_drop_rate),
            "dir_lgbm_skip_drop": float(args.dir_lgbm_skip_drop),
            "dir_lgbm_max_drop": int(args.dir_lgbm_max_drop),
            "dir_lgbm_uniform_drop": bool(args.dir_lgbm_uniform_drop),
            "dir_lgbm_n_jobs": int(args.dir_lgbm_n_jobs),
            "dir_lgbm_verbose": int(args.dir_lgbm_verbose),
            "dir_lgbm_early_stopping_rounds": int(args.dir_lgbm_early_stopping_rounds),
            "dir_lgbm_valid_frac": float(args.dir_lgbm_valid_frac),
            "dir_lgbm_valid_min": int(args.dir_lgbm_valid_min),
            "wf_workers": int(args.wf_workers),
            "sample_decay": float(args.wf_sample_decay),
        }

        # ?? Task preparation (main process, sequential) ??????????????????????
        # Skip logic runs here so workers receive only actionable folds.
        # All df_feat accesses happen here; workers get plain numpy arrays only.
        work_tasks: list[dict] = []
        idx = np.arange(len(df_feat))

        for fold_idx, fold in enumerate(folds):
            if args.wf_mode == "bars":
                tr_start_idx, tr_end_idx, gap_end_idx, os_end_idx = fold
                # Expanding: train from bar 0 to tr_end_idx (ignore tr_start_idx)
                if args.wf_expanding:
                    mask_tr = idx < tr_end_idx
                else:
                    mask_tr = (idx >= tr_start_idx) & (idx < tr_end_idx)
                mask_os = (idx >= gap_end_idx) & (idx < os_end_idx)
                oos_start_pos = gap_end_idx
                keep_train = idx < (oos_start_pos - purge_bars)
            else:
                tr_start, tr_end, gap_end, os_end = fold
                # Expanding: train from the very first date (ignore tr_start)
                if args.wf_expanding:
                    mask_tr = df_feat["date"] < tr_end
                else:
                    mask_tr = (df_feat["date"] >= tr_start) & (df_feat["date"] < tr_end)
                mask_os = (df_feat["date"] >= gap_end) & (df_feat["date"] < os_end)
                oos_pos = np.where(mask_os.to_numpy())[0]
                if oos_pos.size == 0:
                    summary_rows.append(
                        {
                            "fold_idx": fold_idx,
                            "skip_reason": "no_oos_window",
                            "purge_bars": purge_bars,
                        }
                    )
                    continue
                oos_start_pos = int(oos_pos.min())
                keep_train = idx < (oos_start_pos - purge_bars)

            mask_tr_y = mask_tr & df_feat[target_col].notna()
            n_train_before_purge = int(mask_tr_y.sum())
            mask_tr_y = mask_tr_y & keep_train

            if gate_enable and em_round > 0 and gate_pass_mask is not None:
                mask_tr_y_gated = mask_tr_y & gate_pass_mask
                n_train_gated = int(mask_tr_y_gated.sum())
                if n_train_gated < 100:
                    status.line(
                        f"[GATE] fold {fold_idx}: too few gated samples ({n_train_gated}), using all data"
                    )
                else:
                    mask_tr_y = mask_tr_y_gated

            n_train_after_purge = int(mask_tr_y.sum())
            mask_os_y = mask_os & df_feat[target_col].notna()

            tr_ts_min = int(df_feat.loc[mask_tr, "ts_ms"].min()) if mask_tr.any() else None
            tr_ts_max = int(df_feat.loc[mask_tr, "ts_ms"].max()) if mask_tr.any() else None
            os_ts_min = int(df_feat.loc[mask_os, "ts_ms"].min()) if mask_os.any() else None
            os_ts_max = int(df_feat.loc[mask_os, "ts_ms"].max()) if mask_os.any() else None

            if args.wf_mode == "bars":
                fold_dates = {
                    "train_start": tr_ts_min,
                    "train_end": tr_ts_max,
                    "oos_start": os_ts_min,
                    "oos_end": os_ts_max,
                }
            else:
                fold_dates = {
                    "train_start": str(tr_start),
                    "train_end": str(tr_end),
                    "oos_start": str(gap_end),
                    "oos_end": str(os_end),
                }

            if mask_tr_y.sum() < 50 or mask_os_y.sum() < 50:
                summary_rows.append(
                    {
                        "fold_idx": fold_idx,
                        "skip_reason": "insufficient_samples",
                        "n_train_before_purge": n_train_before_purge,
                        "n_train_after_purge": n_train_after_purge,
                        "n_oos": int(mask_os_y.sum()),
                        "purge_bars": purge_bars,
                        "oos_start_pos": oos_start_pos,
                        "tr_start_idx": int(np.where(mask_tr)[0].min()) if mask_tr.any() else None,
                        "tr_end_idx": int(np.where(mask_tr)[0].max()) + 1 if mask_tr.any() else None,
                        "os_start_idx": int(np.where(mask_os)[0].min()) if mask_os.any() else None,
                        "os_end_idx": int(np.where(mask_os)[0].max()) + 1 if mask_os.any() else None,
                        "tr_ts_min": tr_ts_min,
                        "tr_ts_max": tr_ts_max,
                        "os_ts_min": os_ts_min,
                        "os_ts_max": os_ts_max,
                    }
                )
                continue

            X_train_raw = df_feat.loc[mask_tr_y, feature_cols].to_numpy(dtype=float)
            X_oos_raw = df_feat.loc[mask_os_y, feature_cols].to_numpy(dtype=float)
            y_tr = df_feat.loc[mask_tr_y, target_col].to_numpy(dtype=float)
            y_os = df_feat.loc[mask_os_y, target_col].to_numpy(dtype=float)
            ts_ms_oos = df_feat.loc[mask_os_y, "ts_ms"].to_numpy().copy()
            ts_ms_tr = df_feat.loc[mask_tr_y, "ts_ms"].to_numpy().copy()

            context_data: dict = {}
            for _c in _CONTEXT_COLS:
                if _c in df_feat.columns:
                    context_data[_c] = df_feat.loc[mask_os_y, _c].values.copy()

            work_tasks.append(
                {
                    "fold_idx": fold_idx,
                    "X_train_raw": X_train_raw,
                    "X_oos_raw": X_oos_raw,
                    "y_tr": y_tr,
                    "y_os": y_os,
                    "ts_ms_oos": ts_ms_oos,
                    "ts_ms_tr": ts_ms_tr,
                    "context_data": context_data,
                    "fold_dates": fold_dates,
                    "n_train_before_purge": n_train_before_purge,
                    "n_train_after_purge": n_train_after_purge,
                    "oos_start_pos": oos_start_pos,
                    "tr_start_idx": int(np.where(mask_tr)[0].min()) if mask_tr.any() else None,
                    "tr_end_idx": int(np.where(mask_tr)[0].max()) + 1 if mask_tr.any() else None,
                    "os_start_idx": int(np.where(mask_os)[0].min()) if mask_os.any() else None,
                    "os_end_idx": int(np.where(mask_os)[0].max()) + 1 if mask_os.any() else None,
                    "tr_ts_min": tr_ts_min,
                    "tr_ts_max": tr_ts_max,
                    "os_ts_min": os_ts_min,
                    "os_ts_max": os_ts_max,
                    "purge_bars": purge_bars,
                    "model_args": model_args,
                }
            )

        status.done("PREP", f"tasks={len(work_tasks)} skipped={len(folds)-len(work_tasks)}")

        # ?? Parallel / serial execution ????????????????????????????????????
        n_workers = int(args.wf_workers)
        work_results: list[dict] = []

        if n_workers == 1:
            for i, task in enumerate(work_tasks):
                result = _run_fold_worker(task)
                work_results.append(result)
                status.line(
                    f"[WF] fold {task['fold_idx']} done ({i + 1}/{len(work_tasks)})"
                )
        else:
            status.line(
                f"[WF] parallel folds workers={n_workers} total={len(work_tasks)}"
            )
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(_run_fold_worker, task): task["fold_idx"]
                    for task in work_tasks
                }
                done_count = 0
                for fut in as_completed(futures):
                    result = fut.result()  # re-raises any exception from worker
                    work_results.append(result)
                    done_count += 1
                    status.line(
                        f"[WF] fold {result['fold_idx']} done"
                        f" ({done_count}/{len(work_tasks)})"
                    )

        # ?? Result collection (sort by fold_idx to restore temporal order) ??
        work_results.sort(key=lambda r: r["fold_idx"])

        for result in work_results:
            metrics_rows.append(result["metrics"])

            oos_d = result["oos_data"]
            oos_rows = pd.DataFrame({"ts_ms": oos_d["ts_ms"]})
            oos_rows["y_target"] = oos_d["y_target"]
            oos_rows["target_pred"] = oos_d["target_pred"]
            for _c, _v in oos_d["context"].items():
                oos_rows[_c] = _v
            oos_rows["fold_idx"] = oos_d["fold_idx"]
            preds_rows.append(oos_rows)

            summary_rows.append(result["summary"])

        if not metrics_rows:
            raise RuntimeError("no folds produced metrics")

        metrics_df = pd.DataFrame(metrics_rows)
        preds_df = pd.concat(preds_rows, ignore_index=True)
        em_rounds_completed = em_round

        if gate_enable:
            preds_df = _train_gate_lofo(
                preds_df=preds_df,
                df_feat=df_feat,
                feature_cols=feature_cols,
                folds=folds,
                args=args,
                status=status,
            )

            has_gate = preds_df["gate_score"].notna()
            gate_threshold = float("nan")
            preds_df["gate_pass"] = True
            if has_gate.any():
                gate_threshold = float(
                    np.percentile(
                        preds_df.loc[has_gate, "gate_score"],
                        float(args.gate_threshold_pct),
                    )
                )
                preds_df.loc[has_gate, "gate_pass"] = (
                    preds_df.loc[has_gate, "gate_score"] >= gate_threshold
                )

            gate_pass_lookup = preds_df.set_index("ts_ms")["gate_pass"].to_dict()
            gate_pass_mask = (
                df_feat["ts_ms"].map(gate_pass_lookup).fillna(True).to_numpy(dtype=bool)
            )

            y_pred_all = preds_df["target_pred"].to_numpy(dtype=float)
            y_true_all = preds_df["y_target"].to_numpy(dtype=float)
            y_gate_all = y_pred_all * y_true_all
            baseline_spearman = _spearman(y_pred_all, y_true_all)
            baseline_sign_acc = _sign_acc(y_pred_all, y_true_all)
            baseline_mean_pnl = (
                float(np.nanmean(y_gate_all)) if len(y_gate_all) > 0 else float("nan")
            )

            gated_df = preds_df[preds_df["gate_pass"]]
            gated_pred = gated_df["target_pred"].to_numpy(dtype=float)
            gated_true = gated_df["y_target"].to_numpy(dtype=float)
            gated_spearman = _spearman(gated_pred, gated_true)
            gated_sign_acc = _sign_acc(gated_pred, gated_true)
            gated_mean_pnl = (
                float(np.nanmean(gated_pred * gated_true)) if len(gated_pred) > 0 else float("nan")
            )

            n_total = len(preds_df)
            n_gated = int(has_gate.sum())
            n_pass = int(preds_df["gate_pass"].sum())
            pass_rate = float(n_pass / n_total) if n_total > 0 else float("nan")

            gate_sp_map: dict[int, float] = {}
            gated_sp_map: dict[int, float] = {}
            gated_sign_map: dict[int, float] = {}
            pass_rate_map: dict[int, float] = {}

            for f_idx, grp in preds_df.groupby("fold_idx", sort=False):
                y_gate = (
                    np.sign(grp["target_pred"].to_numpy(dtype=float))
                    * grp["y_target"].to_numpy(dtype=float)
                )
                gs = grp["gate_score"].to_numpy(dtype=float)
                if np.isfinite(gs).any():
                    gate_sp = _spearman(gs, y_gate)
                else:
                    gate_sp = float("nan")

                gp_mask = grp["gate_pass"].to_numpy(dtype=bool)
                if gp_mask.sum() > 0:
                    gated_sp = _spearman(
                        grp.loc[gp_mask, "target_pred"].to_numpy(dtype=float),
                        grp.loc[gp_mask, "y_target"].to_numpy(dtype=float),
                    )
                    gated_sign = _sign_acc(
                        grp.loc[gp_mask, "target_pred"].to_numpy(dtype=float),
                        grp.loc[gp_mask, "y_target"].to_numpy(dtype=float),
                    )
                else:
                    gated_sp = float("nan")
                    gated_sign = float("nan")

                gate_sp_map[int(f_idx)] = gate_sp
                gated_sp_map[int(f_idx)] = gated_sp
                gated_sign_map[int(f_idx)] = gated_sign
                pass_rate_map[int(f_idx)] = (
                    float(gp_mask.mean()) if len(gp_mask) > 0 else float("nan")
                )

            metrics_df["gate_spearman"] = metrics_df["fold_idx"].map(gate_sp_map)
            metrics_df["gated_target_spearman"] = metrics_df["fold_idx"].map(gated_sp_map)
            metrics_df["gated_sign_acc"] = metrics_df["fold_idx"].map(gated_sign_map)
            metrics_df["gate_pass_rate"] = metrics_df["fold_idx"].map(pass_rate_map)
            metrics_df["em_round"] = em_round

            em_history.append(
                {
                    "round": int(em_round),
                    "spearman": baseline_spearman,
                    "gated_spearman": gated_spearman,
                    "n_pass": int(n_pass),
                }
            )

            gate_fold_metrics = preds_df.attrs.get("gate_fold_metrics", [])
            if not gate_fold_metrics:
                for f_idx in metrics_df["fold_idx"].tolist():
                    gate_fold_metrics.append(
                        {
                            "fold_idx": int(f_idx),
                            "n_train": None,
                            "gate_spearman": gate_sp_map.get(int(f_idx), float("nan")),
                            "best_iter": None,
                        }
                    )

            gate_metrics = {
                "gate_enabled": True,
                "gate_threshold_pct": float(args.gate_threshold_pct),
                "gate_threshold_value": gate_threshold,
                "em_rounds_completed": int(em_round),
                "per_fold": gate_fold_metrics,
                "overall": {
                    "n_total": int(n_total),
                    "n_gated": int(n_gated),
                    "n_pass": int(n_pass),
                    "pass_rate": pass_rate,
                    "baseline_spearman": baseline_spearman,
                    "gated_spearman": gated_spearman,
                    "baseline_sign_acc": baseline_sign_acc,
                    "gated_sign_acc": gated_sign_acc,
                    "baseline_mean_pnl": baseline_mean_pnl,
                    "gated_mean_pnl": gated_mean_pnl,
                },
                "em_history": em_history,
            }

            status.line(
                f"[GATE] EM round {em_round}: "
                f"pass={n_pass}/{n_total} ({pass_rate:.1%}) "
                f"sp_gated={gated_spearman:.4f} threshold={gate_threshold:.4f}"
            )

            if prev_sp_gated is not None:
                sp_delta = abs(gated_spearman - prev_sp_gated)
                if sp_delta < 0.005:
                    status.line(
                        f"[GATE] EM converged at round {em_round} (delta={sp_delta:.4f})"
                    )
                    break
            prev_sp_gated = gated_spearman

            if em_round >= gate_em_rounds:
                break
        else:
            break

    if metrics_df is None or preds_df is None:
        raise RuntimeError("no folds produced metrics")

    if gate_enable and gate_metrics is not None:
        gate_metrics["em_rounds_completed"] = int(em_rounds_completed)
    _save_results(
        outdir=args.outdir,
        metrics_df=metrics_df,
        preds_df=preds_df,
        summary_rows=summary_rows,
        gate_enable=gate_enable,
        gate_metrics=gate_metrics,
        args=args,
        feature_cols=feature_cols,
        df_feat=df_feat,
        target_col=target_col,
    )
    # ── Structure B: Exit model ──────────────────────────────────────────────
    if args.target_mode == "rt3_event":
        status.update(
            "EXIT",
            "extracting tick paths (2nd pass, entry_pred direction)...",
            force=True,
        )

        preds_map = preds_df[["ts_ms", "target_pred", "fold_idx"]].copy()
        preds_map = preds_map.dropna(subset=["ts_ms"])
        preds_map = preds_map.drop_duplicates(subset=["ts_ms"], keep="last")
        preds_map = preds_map.rename(columns={"target_pred": "entry_pred"})
        merged = df_feat[["ts_ms"]].merge(preds_map, on="ts_ms", how="left", sort=False)
        if len(merged) != len(df_feat):
            raise RuntimeError("entry preds alignment failed (row count mismatch)")
        entry_preds_arr = merged["entry_pred"].to_numpy(dtype=float)
        df_feat["fold_idx"] = merged["fold_idx"].to_numpy()

        tick_df = _exit_extract_tick_paths(
            df_feat=df_feat,
            zip_paths=all_paths,
            entry_preds=entry_preds_arr,
            cur_vol_vb_bp_arr=df_feat["cur_vol_vb_bp"].to_numpy(dtype=float),
            n_samples=int(args.exit_n_samples),
            lag_interval_sec=float(args.exit_lag_interval_sec),
            n_lags=int(args.exit_n_lags),
            recent_window=int(args.exit_recent_window),
        )
        status.done("EXIT", f"tick_df rows={len(tick_df):,}")
        # ── PnL vol-normalization (path non-stationarity) ──
        if not tick_df.empty:
            _vol_scale = tick_df["entry_vol"].to_numpy(dtype=float) * 1e-4
            _vol_scale = np.maximum(_vol_scale, 1.0 * 1e-4)
            _pnl_norm_cols = [
                "pnl_peak",
                "pnl_from_peak",
                "pnl_velocity",
                "pnl_acceleration",
                "exit_target",
            ]
            _pnl_norm_cols += [
                c for c in tick_df.columns
                if c.startswith("current_pnl_lag_")
                or c.startswith("pnl_from_peak_lag_")
            ]
            for _col in _pnl_norm_cols:
                if _col in tick_df.columns:
                    tick_df[_col] = (
                        tick_df[_col].to_numpy(dtype=float) / _vol_scale
                    )
        # ── entry_pred direction alignment ──
        if not tick_df.empty:
            tick_df["entry_pred"] = (
                tick_df["entry_pred"].to_numpy(dtype=float)
                * tick_df["entry_dir"].to_numpy(dtype=float)
            )
        if tick_df.empty:
            status.line("[EXIT] tick_df empty; skipping exit model")
        else:
            status.update("EXIT", "training exit model WF...", force=True)
            exit_models = _exit_train_wf(tick_df, folds, args)

            oos_preds = np.full(len(tick_df), np.nan, dtype=float)
            exit_exclude = {
                "event_id",
                "fold_idx",
                "elapsed_sec",
                "exit_target",
                "sample_weight",
                "entry_price",
                "current_pnl",
                "cum_imb_signed",
                "entry_dir",
                "entry_vol",
            }
            feat_cols = [c for c in tick_df.columns if c not in exit_exclude]
            for k, model in enumerate(exit_models):
                if model is None:
                    continue
                mask = tick_df["fold_idx"] == k
                if not mask.any():
                    continue
                oos_preds[mask] = model.predict(
                    tick_df.loc[mask, feat_cols]
                )

            exit_metrics = _exit_evaluate(tick_df, oos_preds, args)

            tick_df = tick_df.copy()
            tick_df["exit_pred"] = oos_preds
            exit_feature_cols = [
                c
                for c in tick_df.columns
                if c not in exit_exclude and c not in {"entry_pred", "exit_pred"}
            ]
            exit_cols = (
                [
                    "event_id",
                    "fold_idx",
                    "elapsed_sec",
                    "exit_target",
                    "exit_pred",
                    "sample_weight",
                    "entry_pred",
                ]
                + exit_feature_cols
            )
            tick_df.to_csv(
                os.path.join(args.outdir, "exit_preds.csv"),
                index=False,
                columns=exit_cols,
            )

            with open(os.path.join(args.outdir, "exit_metrics.json"), "w") as f:
                json.dump(exit_metrics, f, indent=2)

            status.done(
                "EXIT",
                f"mean_improvement={exit_metrics['mean_improvement_bp']:.2f}bp "
                f"pct_improved={exit_metrics['pct_events_improved']:.1%}",
            )

def _exit_fold_worker(task: dict) -> dict:
    """
    Module-level exit fold worker; picklable for ProcessPoolExecutor.
    Trains one LGBMRegressor fold (huber objective) and returns the trained model.
    """
    import lightgbm as lgb

    fold_idx = task["fold_idx"]
    X_tr = task["X_tr"]
    y_tr = task["y_tr"]
    w_tr = task["w_tr"]
    X_val = task["X_val"]
    y_val = task["y_val"]
    w_val = task["w_val"]
    params = task["lgbm_params"]
    es_rounds = task["early_stopping_rounds"]
    seed = task["seed"]

    model = lgb.LGBMRegressor(
        objective="huber",
        boosting_type="gbdt",
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_samples=params["min_child_samples"],
        feature_fraction=params["feature_fraction"],
        subsample=params["subsample"],
        subsample_freq=params["subsample_freq"],
        random_state=seed,
        n_jobs=params["n_jobs"],
        verbose=-1,
    )

    callbacks = [lgb.log_evaluation(period=-1)]
    if es_rounds > 0:
        callbacks.append(lgb.early_stopping(es_rounds, verbose=False))

    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="huber",
        callbacks=callbacks,
    )

    return {"fold_idx": fold_idx, "model": model}


def _exit_nearest_indices(sorted_vals: np.ndarray, target_vals: np.ndarray) -> np.ndarray:
    sorted_vals = np.asarray(sorted_vals, dtype=float)
    target_vals = np.asarray(target_vals, dtype=float)
    if sorted_vals.size == 0:
        return np.empty_like(target_vals, dtype=np.int64)
    idx = np.searchsorted(sorted_vals, target_vals, side="left")
    idx = np.clip(idx, 0, len(sorted_vals) - 1)
    left = np.maximum(idx - 1, 0)
    right = idx
    left_dist = np.abs(target_vals - sorted_vals[left])
    right_dist = np.abs(sorted_vals[right] - target_vals)
    choose_right = right_dist < left_dist
    return np.where(choose_right, right, left).astype(np.int64)


def _exit_finalize_event_arrays(
    ev: dict,
    f_elapsed_sec: list,
    f_current_pnl: list,
    f_pnl_peak: list,
    f_pnl_from_peak: list,
    f_cum_imb_signed: list,
    f_cum_buy_vol: list,
    f_cum_sell_vol: list,
    f_tick_gap_sec: list,
    f_price: list,
    n_samples: int,
    n_lags: int,
    lag_interval_sec: float,
    recent_window: int,
) -> pd.DataFrame | None:
    n_raw = len(f_elapsed_sec)
    if n_raw < 2:
        return None

    elapsed = np.array(f_elapsed_sec, dtype=float)
    current_pnl = np.array(f_current_pnl, dtype=float)
    pnl_peak = np.array(f_pnl_peak, dtype=float)
    pnl_from_peak = np.array(f_pnl_from_peak, dtype=float)
    cum_imb_signed = np.array(f_cum_imb_signed, dtype=float)
    cum_buy_vol = np.array(f_cum_buy_vol, dtype=float)
    cum_sell_vol = np.array(f_cum_sell_vol, dtype=float)
    tick_gap = np.array(f_tick_gap_sec, dtype=float)
    prices = np.array(f_price, dtype=float)

    sort_idx = np.argsort(elapsed, kind="quicksort")
    elapsed = elapsed[sort_idx]
    current_pnl = current_pnl[sort_idx]
    pnl_peak = pnl_peak[sort_idx]
    pnl_from_peak = pnl_from_peak[sort_idx]
    cum_imb_signed = cum_imb_signed[sort_idx]
    cum_buy_vol = cum_buy_vol[sort_idx]
    cum_sell_vol = cum_sell_vol[sort_idx]
    tick_gap = tick_gap[sort_idx]
    prices = prices[sort_idx]

    if len(np.unique(elapsed)) < min(10, n_samples // 10):
        return None
    if not np.all(np.diff(elapsed) >= 0):
        raise ValueError("event ticks must be non-decreasing in elapsed_sec")
    max_elapsed = float(elapsed[-1])
    if not np.isfinite(max_elapsed) or max_elapsed <= 0:
        return None

    sample_times = np.linspace(0.0, max_elapsed, n_samples)
    nearest_idx = _exit_nearest_indices(elapsed, sample_times)

    elapsed_s = elapsed[nearest_idx]
    current_pnl_s = current_pnl[nearest_idx]
    pnl_peak_s = pnl_peak[nearest_idx]
    pnl_from_peak_s = pnl_from_peak[nearest_idx]
    cum_imb_signed_s = cum_imb_signed[nearest_idx]
    cum_buy_vol_s = cum_buy_vol[nearest_idx]
    cum_sell_vol_s = cum_sell_vol[nearest_idx]
    prices_s = prices[nearest_idx]

    if not np.all(np.diff(elapsed_s) >= 0):
        raise ValueError(
            f"event_id {ev['event_id']} sampled elapsed_sec not monotonic"
        )
    if not np.all(np.diff(elapsed_s) > 0):
        return None

    N = len(elapsed_s)

    lag_keys = range(1, n_lags + 1)
    lag_sec_arr = np.array(
        [float(k) * lag_interval_sec for k in lag_keys], dtype=float
    )
    pnl_lag = np.zeros((n_lags, N), dtype=float)
    fpeak_lag = np.zeros((n_lags, N), dtype=float)
    imb_lag = np.zeros((n_lags, N), dtype=float)
    lag_valid = np.zeros((n_lags, N), dtype=float)

    for ki, lag_sec in enumerate(lag_sec_arr):
        target_times = elapsed_s - lag_sec
        li = _exit_nearest_indices(elapsed_s, target_times)
        valid = elapsed_s >= lag_sec
        lag_valid[ki] = valid.astype(float)
        pnl_lag[ki] = np.where(valid, current_pnl_s[li], 0.0)
        fpeak_lag[ki] = np.where(valid, pnl_from_peak_s[li], 0.0)
        imb_lag[ki] = np.where(valid, cum_imb_signed_s[li], 0.0)

    lag1_valid_mask = lag_valid[0] > 0.0
    lag1_pnl = pnl_lag[0]
    lag1_imb = imb_lag[0]

    pnl_velocity = np.zeros(N, dtype=float)
    pnl_velocity[lag1_valid_mask] = (
        (current_pnl_s[lag1_valid_mask] - lag1_pnl[lag1_valid_mask])
        / lag_interval_sec
    )

    pnl_acceleration = np.zeros(N, dtype=float)
    if n_lags >= 2:
        lag2_valid_mask = lag_valid[1] > 0.0
        lag2_pnl = pnl_lag[1]
        accel_mask = lag1_valid_mask & lag2_valid_mask
        pnl_acceleration[accel_mask] = pnl_velocity[accel_mask] - (
            (lag1_pnl[accel_mask] - lag2_pnl[accel_mask])
            / lag_interval_sec
        )

    imb_velocity = np.zeros(N, dtype=float)
    imb_velocity[lag1_valid_mask] = (
        (cum_imb_signed_s[lag1_valid_mask] - lag1_imb[lag1_valid_mask])
        / lag_interval_sec
    )

    buy_vol_recent = np.zeros(N, dtype=float)
    sell_vol_recent = np.zeros(N, dtype=float)
    if lag1_valid_mask.any():
        lag1_idx = _exit_nearest_indices(
            elapsed_s, elapsed_s - lag_interval_sec
        )
        buy_vol_recent[lag1_valid_mask] = (
            cum_buy_vol_s[lag1_valid_mask]
            - cum_buy_vol_s[lag1_idx[lag1_valid_mask]]
        )
        sell_vol_recent[lag1_valid_mask] = (
            cum_sell_vol_s[lag1_valid_mask]
            - cum_sell_vol_s[lag1_idx[lag1_valid_mask]]
        )
    total_vol_recent = buy_vol_recent + sell_vol_recent
    buy_vol_rate_recent = np.zeros(N, dtype=float)
    imb_rate_recent = np.zeros(N, dtype=float)
    valid_recent = total_vol_recent > 0.0
    buy_vol_rate_recent[valid_recent] = (
        buy_vol_recent[valid_recent]
        / (total_vol_recent[valid_recent] + EPS)
    )
    imb_rate_recent[valid_recent] = (
        (buy_vol_recent[valid_recent] - sell_vol_recent[valid_recent])
        / (total_vol_recent[valid_recent] + EPS)
    )

    roll_w = min(recent_window, N)

    def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
        cs = np.cumsum(arr)
        result = cs.copy()
        result[w:] = cs[w:] - cs[:-w]
        counts = np.minimum(np.arange(1, len(arr) + 1), w)
        return result / counts

    price_diff = np.diff(prices_s, prepend=prices_s[0])
    sign_match = (np.sign(price_diff) == np.sign(ev["entry_dir"])).astype(float)
    n_up_ticks_pct_last = _rolling_mean(sign_match, roll_w)

    # ── Path phase features ──
    # pnl_realized_ratio: fraction of peak profit retained (self-normalizing)
    pnl_realized_ratio = np.where(
        pnl_peak_s > 1e-10,
        current_pnl_s / pnl_peak_s,
        0.0,
    )

    # time_since_peak_sec: seconds elapsed since pnl_peak was last updated
    peak_new = np.empty(N, dtype=bool)
    peak_new[0] = True
    peak_new[1:] = pnl_peak_s[1:] > pnl_peak_s[:-1]
    last_peak_idx = np.maximum.accumulate(
        np.where(peak_new, np.arange(N), 0)
    )
    time_since_peak_sec = elapsed_s - elapsed_s[last_peak_idx]

    # ?? imb_decay: order flow exhaustion ??
    # ??? 3???? ??/?? ???? ?? ?? ??
    third = max(N // 3, 1)
    imb_early = cum_imb_signed_s[min(third, N - 1)] - cum_imb_signed_s[0]
    elapsed_early = elapsed_s[min(third, N - 1)] - elapsed_s[0]
    imb_late = cum_imb_signed_s[N - 1] - cum_imb_signed_s[max(N - third, 0)]
    elapsed_late = elapsed_s[N - 1] - elapsed_s[max(N - third, 0)]

    rate_early = imb_early / max(elapsed_early, 1e-6)
    rate_late = imb_late / max(elapsed_late, 1e-6)
    # ???? ??? ? ?? ??? broadcast
    _imb_decay_val = abs(rate_late) / (abs(rate_early) + 1e-9)
    imb_decay = np.full(N, min(_imb_decay_val, 10.0), dtype=float)

    # ?? vol_surprise: realized vs expected volatility ??
    # ?? ? ?? ?????? ???? / entry_vol
    _entry_vol = ev.get("entry_vol", 0.0)
    if N >= 3 and _entry_vol > 0:
        log_rets = np.diff(np.log(np.maximum(prices_s, 1e-10)))
        realized_vol = float(np.std(log_rets)) * 1e4  # bp ???
        _vol_surprise_val = realized_vol / max(_entry_vol, 1e-4)
    else:
        _vol_surprise_val = 1.0
    vol_surprise = np.full(N, _vol_surprise_val, dtype=float)


    # ?? Continuation value target: remaining_upside ??
    # remaining_upside(t) = max(pnl[t:T]) - pnl(t)
    # 0?? ??? ??(?? ??), ??? ?? ?? ?? ??
    remaining_max = np.maximum.accumulate(current_pnl_s[::-1])[::-1]
    remaining_upside = remaining_max - current_pnl_s
    remaining_upside = np.maximum(remaining_upside, 0.0)  # ?? ??

    out: dict[str, np.ndarray] = {
        "event_id": np.full(N, ev["event_id"], dtype=np.int64),
        "fold_idx": np.full(N, ev["fold_idx"], dtype=np.int64),
        "elapsed_sec": elapsed_s,
        "current_pnl": current_pnl_s,
        "pnl_peak": pnl_peak_s,
        "pnl_from_peak": pnl_from_peak_s,
        "cum_imb_signed": cum_imb_signed_s,
        "entry_pred": np.full(N, ev["entry_pred"], dtype=float),
        "entry_price": np.full(N, ev["entry_price"], dtype=float),
        "entry_dir": np.full(N, ev.get("entry_dir", 0.0), dtype=float),
        "entry_vol": np.full(N, ev.get("entry_vol", 0.0), dtype=float),
    }
    for ki, lag_sec in enumerate(lag_sec_arr):
        lbl = f"{lag_sec:.0f}s"
        out[f"current_pnl_lag_{lbl}"] = pnl_lag[ki]
        out[f"pnl_from_peak_lag_{lbl}"] = fpeak_lag[ki]
        out[f"cum_imb_signed_lag_{lbl}"] = imb_lag[ki]
        out[f"lag_valid_{lbl}"] = lag_valid[ki]
    out["n_up_ticks_pct_last20"] = n_up_ticks_pct_last
    out["pnl_velocity"] = pnl_velocity
    out["pnl_acceleration"] = pnl_acceleration
    out["imb_velocity"] = imb_velocity
    out["buy_vol_rate_recent"] = buy_vol_rate_recent
    out["imb_rate_recent"] = imb_rate_recent
    out["pnl_realized_ratio"] = pnl_realized_ratio
    out["time_since_peak_sec"] = time_since_peak_sec
    out["imb_decay"] = imb_decay
    out["vol_surprise"] = vol_surprise
    out["exit_target"] = remaining_upside
    out["sample_weight"] = np.full(N, 1.0 / float(n_samples), dtype=float)

    return pd.DataFrame(out)


def _exit_extract_tick_paths(
    df_feat: pd.DataFrame,
    zip_paths: list[str],
    entry_preds: np.ndarray,
    cur_vol_vb_bp_arr: np.ndarray,
    n_samples: int,
    lag_interval_sec: float,
    n_lags: int,
    recent_window: int,
) -> pd.DataFrame:
    n_samples = int(n_samples)
    n_lags = int(n_lags)
    lag_interval_sec = float(lag_interval_sec)
    recent_window = int(recent_window)
    if n_samples <= 0:
        raise ValueError("exit_n_samples must be > 0")
    if n_lags < 1:
        raise ValueError("exit_n_lags must be >= 1")
    if lag_interval_sec <= 0:
        raise ValueError("exit_lag_interval_sec must be > 0")
    if recent_window < 1:
        raise ValueError("exit_recent_window must be >= 1")
    if len(entry_preds) != len(df_feat):
        raise ValueError("entry_preds length must match df_feat")
    if len(cur_vol_vb_bp_arr) != len(df_feat):
        raise ValueError("cur_vol_vb_bp_arr length must match df_feat")

    rt3_on = df_feat["rt3_on"].to_numpy(dtype=bool)
    rt3_off = df_feat["rt3_off"].to_numpy(dtype=bool)
    y_event = pd.to_numeric(df_feat["y_event"], errors="coerce").to_numpy(dtype=float)
    ts_ms_arr = pd.to_numeric(df_feat["ts_ms"], errors="coerce").to_numpy(dtype=np.int64)
    close_arr = pd.to_numeric(df_feat["close"], errors="coerce").to_numpy(dtype=float)
    if "fold_idx" in df_feat.columns:
        fold_idx_arr = pd.to_numeric(df_feat["fold_idx"], errors="coerce").to_numpy(dtype=float)
    else:
        fold_idx_arr = np.full(len(df_feat), np.nan, dtype=float)

    n = len(df_feat)
    next_off = np.full(n, n, dtype=np.int64)
    if n > 0 and rt3_off[n - 1]:
        next_off[n - 1] = n - 1
    for i in range(n - 2, -1, -1):
        if rt3_off[i]:
            next_off[i] = i
        else:
            next_off[i] = next_off[i + 1]

    events: list[dict] = []
    event_indices = np.where(rt3_on & np.isfinite(y_event))[0]
    for event_id, t in enumerate(event_indices):
        if t + 1 >= n:
            continue
        t_exit = int(next_off[t + 1])
        if t_exit >= n:
            continue
        entry_price = float(close_arr[t])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        entry_pred = float(entry_preds[t])
        if not np.isfinite(entry_pred):
            continue
        entry_dir = float(np.sign(entry_preds[t]))
        if entry_dir == 0.0:
            continue
        fold_idx = float(fold_idx_arr[t])
        if not np.isfinite(fold_idx):
            continue
        ts_entry = int(ts_ms_arr[t])
        ts_exit = int(ts_ms_arr[t_exit])
        if ts_exit <= ts_entry:
            continue
        # ?? ??? ?? ? ??? VB? entry_pred ?? ??
        # bar t (??) ~ bar t_exit (??) ?? ?? VB
        vb_indices = list(range(t, t_exit + 1))
        _valid_vb = [
            (int(ts_ms_arr[i]), float(entry_preds[i]) * entry_dir)
            for i in vb_indices
            if i < len(entry_preds) and np.isfinite(entry_preds[i])
        ]

        events.append(
            {
                "event_id": int(event_id),
                "fold_idx": int(fold_idx),
                "ts_entry_ms": ts_entry,
                "ts_exit_ms": ts_exit,
                "entry_price": entry_price,
                "entry_dir": entry_dir,
                "entry_pred": entry_pred,
                "entry_vol": (
                    float(cur_vol_vb_bp_arr[t])
                    if t < len(cur_vol_vb_bp_arr)
                    and np.isfinite(cur_vol_vb_bp_arr[t])
                    else 0.0
                ),
                "vb_ts_ms": [x[0] for x in _valid_vb],
                "vb_preds": [x[1] for x in _valid_vb],
            }
        )

    if not events:
        return pd.DataFrame()

    events.sort(key=lambda e: e["ts_entry_ms"])
    start_ms = min(e["ts_entry_ms"] for e in events)
    end_ms = max(e["ts_exit_ms"] for e in events) + 1

    # ── Phase 2: zip filtering ──────────────────────────────────────────
    # Only read zips whose date range overlaps [start_ms, end_ms).
    filtered_zips: list[str] = []
    for zp in zip_paths:
        try:
            zd_start, zd_end = _zip_date_ms_range(zp)
        except ValueError:
            filtered_zips.append(zp)
            continue
        if zd_end > start_ms and zd_start < end_ms:
            filtered_zips.append(zp)

    # ── Phase 3: load and concatenate zip arrays ─────────────────────
    _zip_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    _MAX_CACHE = 3

    def _get_zip_arrays(zp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if zp not in _zip_cache:
            if len(_zip_cache) >= _MAX_CACHE:
                oldest = next(iter(_zip_cache))
                del _zip_cache[oldest]
            _zip_cache[zp] = _read_zip_arrays(zp)
        return _zip_cache[zp]

    # ── Phase 4: build date→zip lookup for events ────────────────────
    zip_by_date: list[tuple[int, int, str]] = []
    for zp in filtered_zips:
        try:
            zd_start, zd_end = _zip_date_ms_range(zp)
        except ValueError:
            zd_start, zd_end = 0, int(2e15)
        zip_by_date.append((zd_start, zd_end, zp))
    zip_by_date.sort(key=lambda x: x[0])

    def _find_zips_for_range(t_start: int, t_end: int) -> list[str]:
        result = []
        for zd_start, zd_end, zp in zip_by_date:
            if zd_start > t_end:
                break
            if zd_end > t_start:
                result.append(zp)
        return result

    # ── Phase 5: vectorized per-event processing ─────────────────────
    sampled_frames: list[pd.DataFrame] = []

    for ev in events:
        ts_entry = ev["ts_entry_ms"]
        ts_exit = ev["ts_exit_ms"]
        entry_price = ev["entry_price"]
        entry_dir = ev["entry_dir"]

        ev_zips = _find_zips_for_range(ts_entry, ts_exit)
        if not ev_zips:
            continue

        chunks_ts = []
        chunks_price = []
        chunks_qty = []
        chunks_ibm = []
        for zp in ev_zips:
            z_ts, z_price, z_qty, z_ibm = _get_zip_arrays(zp)
            if len(z_ts) == 0:
                continue
            chunks_ts.append(z_ts)
            chunks_price.append(z_price)
            chunks_qty.append(z_qty)
            chunks_ibm.append(z_ibm)

        if not chunks_ts:
            continue

        if len(chunks_ts) == 1:
            all_ts = chunks_ts[0]
            all_price = chunks_price[0]
            all_qty = chunks_qty[0]
            all_ibm = chunks_ibm[0]
        else:
            all_ts = np.concatenate(chunks_ts)
            all_price = np.concatenate(chunks_price)
            all_qty = np.concatenate(chunks_qty)
            all_ibm = np.concatenate(chunks_ibm)

        i_start = int(np.searchsorted(all_ts, ts_entry, side="left"))
        i_end = int(np.searchsorted(all_ts, ts_exit, side="right"))
        if i_end <= i_start:
            continue

        ts_arr = all_ts[i_start:i_end]
        price_arr = all_price[i_start:i_end]
        qty_arr = all_qty[i_start:i_end]
        ibm_arr = all_ibm[i_start:i_end]

        n_ticks = len(ts_arr)
        if n_ticks < 2:
            continue
        if entry_price <= 0.0:
            continue

        elapsed = (ts_arr.astype(np.float64) - float(ts_entry)) / 1000.0
        log_ret = np.log(price_arr / entry_price)
        current_pnl = log_ret * entry_dir
        pnl_peak = np.maximum.accumulate(current_pnl)
        pnl_from_peak = current_pnl - pnl_peak

        buy_qty = np.where(~ibm_arr, qty_arr, 0.0)
        sell_qty = np.where(ibm_arr, qty_arr, 0.0)
        cum_buy_vol = np.cumsum(buy_qty)
        cum_sell_vol = np.cumsum(sell_qty)
        cum_total_vol = cum_buy_vol + cum_sell_vol
        cum_imb_raw = (cum_buy_vol - cum_sell_vol) / np.maximum(cum_total_vol, EPS)
        cum_imb_signed = cum_imb_raw * entry_dir

        tick_gap = np.empty(n_ticks, dtype=float)
        tick_gap[0] = 0.0
        tick_gap[1:] = np.diff(elapsed)

        result = _exit_finalize_event_arrays(
            ev,
            elapsed,
            current_pnl,
            pnl_peak,
            pnl_from_peak,
            cum_imb_signed,
            cum_buy_vol,
            cum_sell_vol,
            tick_gap,
            price_arr,
            n_samples,
            n_lags,
            lag_interval_sec,
            recent_window,
        )
        if result is None:
            continue

        # ?? path_pred ?? ?? ??????????????????????????????
        _vb_ts = ev.get("vb_ts_ms", [])
        _vb_preds = ev.get("vb_preds", [])
        _N = len(result)

        if len(_vb_ts) >= 1 and len(_vb_preds) >= 1:
            _vb_elapsed = np.array(
                [(t - ev["ts_entry_ms"]) / 1000.0 for t in _vb_ts],
                dtype=float,
            )
            _vb_pred_arr = np.array(_vb_preds, dtype=float)
            _sample_elapsed = result["elapsed_sec"].to_numpy(dtype=float)

            _vb_idx = (
                np.searchsorted(_vb_elapsed, _sample_elapsed, side="right")
                - 1
            )
            _vb_idx = np.clip(_vb_idx, 0, len(_vb_pred_arr) - 1)

            _path_pred_current = _vb_pred_arr[_vb_idx]
            _path_pred_initial = _vb_preds[0]

            result["path_pred_current"] = _path_pred_current
            result["path_pred_delta"] = _path_pred_current - _path_pred_initial
            result["path_pred_ratio"] = _path_pred_current / (
                abs(_path_pred_initial) + 1e-9
            )

            if len(_vb_pred_arr) >= 2:
                _recent_n = min(3, len(_vb_pred_arr))
                _recent_preds = _vb_pred_arr[-_recent_n:]
                _x = np.arange(_recent_n, dtype=float)
                _slope = float(np.polyfit(_x, _recent_preds, 1)[0])
            else:
                _slope = 0.0
            result["path_pred_slope"] = np.full(_N, _slope, dtype=float)
            result["path_n_vb_completed"] = np.full(
                _N, float(len(_vb_pred_arr)), dtype=float
            )
        else:
            _initial = abs(ev.get("entry_pred", 0.0))
            result["path_pred_current"] = np.full(_N, _initial, dtype=float)
            result["path_pred_delta"] = np.zeros(_N, dtype=float)
            result["path_pred_ratio"] = np.ones(_N, dtype=float)
            result["path_pred_slope"] = np.zeros(_N, dtype=float)
            result["path_n_vb_completed"] = np.zeros(_N, dtype=float)

        sampled_frames.append(result)

    _zip_cache.clear()

    if not sampled_frames:
        return pd.DataFrame()
    return pd.concat(sampled_frames, ignore_index=True)


def _exit_train_wf(
    tick_df: pd.DataFrame,
    fold_boundaries: list[dict],
    args,
) -> list[object]:
    if lgb is None:
        raise RuntimeError("lightgbm not installed; required for exit model")

    EXCLUDE_COLS = {
        "event_id",
        "fold_idx",
        "elapsed_sec",
        "exit_target",
        "sample_weight",
        "entry_price",
        "current_pnl",
        "cum_imb_signed",
        "entry_dir",
        "entry_vol",
    }
    feature_cols = [c for c in tick_df.columns if c not in EXCLUDE_COLS]
    n_folds = len(fold_boundaries)
    valid_frac = float(args.exit_valid_frac)
    n_workers = int(args.wf_workers)

    lgbm_n_jobs = 1 if n_workers > 1 else -1

    lgbm_params = {
        "n_estimators": int(args.exit_lgbm_n_estimators),
        "num_leaves": int(args.exit_lgbm_num_leaves),
        "max_depth": int(args.exit_lgbm_max_depth),
        "learning_rate": float(args.exit_lgbm_learning_rate),
        "min_child_samples": int(args.exit_lgbm_min_data_in_leaf),
        "feature_fraction": float(args.exit_lgbm_feature_fraction),
        "subsample": float(args.exit_lgbm_bagging_fraction),
        "subsample_freq": int(args.exit_lgbm_bagging_freq),
        "n_jobs": lgbm_n_jobs,
    }
    es_rounds = int(args.exit_lgbm_early_stopping_rounds)
    seed = int(args.seed)

    X_all = tick_df[feature_cols].to_numpy(dtype=float)
    y_all = tick_df["exit_target"].to_numpy(dtype=float)
    event_id_all = tick_df["event_id"].to_numpy(dtype=int)
    w_all = tick_df["sample_weight"].to_numpy(dtype=float)
    fold_idx_arr = tick_df["fold_idx"].to_numpy(dtype=int)

    tasks: list[dict] = []
    skipped_folds: list[int] = []

    for k in range(n_folds):
        if args.exit_wf_expanding:
            train_mask = fold_idx_arr < k
        else:
            if k == 0:
                skipped_folds.append(k)
                continue
            train_mask = fold_idx_arr == (k - 1)

        oos_mask = fold_idx_arr == k

        if train_mask.sum() == 0 or oos_mask.sum() == 0:
            skipped_folds.append(k)
            continue

        X_tr_full = X_all[train_mask]
        y_tr_full = y_all[train_mask]
        w_tr_full = w_all[train_mask]
        eid_tr_full = event_id_all[train_mask]
        n_train = len(y_tr_full)

        if n_train < 2:
            skipped_folds.append(k)
            continue

        # ?? event-based validation split ??
        # ??? ??? ??: ??? valid_frac ??? ???? validation??
        unique_eids = np.unique(eid_tr_full)
        n_val_events = max(1, int(math.floor(len(unique_eids) * valid_frac)))
        if n_val_events >= len(unique_eids):
            n_val_events = len(unique_eids) - 1
        val_event_set = set(unique_eids[-n_val_events:].tolist())
        val_mask_local = np.array(
            [eid in val_event_set for eid in eid_tr_full], dtype=bool
        )
        tr_mask_local = ~val_mask_local

        if tr_mask_local.sum() == 0 or val_mask_local.sum() == 0:
            skipped_folds.append(k)
            continue

        tasks.append(
            {
                "fold_idx": k,
                "X_tr": X_tr_full[tr_mask_local],
                "y_tr": y_tr_full[tr_mask_local],
                "w_tr": w_tr_full[tr_mask_local],
                "X_val": X_tr_full[val_mask_local],
                "y_val": y_tr_full[val_mask_local],
                "w_val": w_tr_full[val_mask_local],
                "lgbm_params": lgbm_params,
                "early_stopping_rounds": es_rounds,
                "seed": seed,
            }
        )

    results: list[dict] = []

    if n_workers == 1:
        for task in tasks:
            results.append(_exit_fold_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_exit_fold_worker, task): task["fold_idx"]
                for task in tasks
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda r: r["fold_idx"])
    result_by_fold = {r["fold_idx"]: r["model"] for r in results}

    models: list[object] = []
    for k in range(n_folds):
        if k in skipped_folds:
            models.append(None)
        elif k in result_by_fold:
            models.append(result_by_fold[k])
        else:
            models.append(None)

    return models


def _exit_evaluate(
    tick_df: pd.DataFrame,
    oos_preds: np.ndarray,
    args,
) -> dict:
    """
    Evaluate exit model: predicted remaining_upside ? exit when below ?.
    Compare model-driven exit PnL vs rule-based exit PnL.
    """
    if len(oos_preds) != len(tick_df):
        raise ValueError("oos_preds must align with tick_df")
    df = tick_df.copy()
    df["exit_pred"] = oos_preds

    exit_alpha = float(args.exit_threshold)
    model_pnls: list[float] = []
    rule_pnls: list[float] = []
    improvements_bp: list[float] = []
    exit_elapsed_pct: list[float] = []

    for event_id, grp in df.groupby("event_id", sort=False):
        grp = grp.sort_values("elapsed_sec")
        preds = grp["exit_pred"].to_numpy(dtype=float)
        preds = np.nan_to_num(preds, nan=np.inf)
        pnl = grp["current_pnl"].to_numpy(dtype=float)
        elapsed_sec_arr = grp["elapsed_sec"].to_numpy(dtype=float)
        if len(pnl) == 0:
            continue
        hit = preds <= exit_alpha
        if np.any(hit):
            exit_idx = int(np.argmax(hit))
        else:
            exit_idx = len(pnl) - 1
        exit_pnl_model = float(pnl[exit_idx])
        exit_pnl_rule = float(pnl[-1])
        model_pnls.append(exit_pnl_model)
        rule_pnls.append(exit_pnl_rule)
        improvements_bp.append((exit_pnl_model - exit_pnl_rule) * 1e4)
        _max_el = float(elapsed_sec_arr[-1]) if len(elapsed_sec_arr) > 0 else 1.0
        exit_elapsed_pct.append(
            float(elapsed_sec_arr[exit_idx] / max(_max_el, 1e-9))
        )

    n_events = len(model_pnls)
    model_mean = float(np.mean(model_pnls)) if model_pnls else float("nan")
    rule_mean = float(np.mean(rule_pnls)) if rule_pnls else float("nan")
    model_std = float(np.std(model_pnls)) if len(model_pnls) > 1 else float("nan")
    rule_std = float(np.std(rule_pnls)) if len(rule_pnls) > 1 else float("nan")
    model_sharpe = (
        model_mean / model_std if model_std and model_std > 0 else float("nan")
    )
    rule_sharpe = (
        rule_mean / rule_std if rule_std and rule_std > 0 else float("nan")
    )
    pct_improved = (
        float(np.mean(np.array(improvements_bp) > 0.0))
        if improvements_bp
        else float("nan")
    )
    mean_improvement = (
        float(np.mean(improvements_bp)) if improvements_bp else float("nan")
    )
    exit_elapsed_pct_mean = (
        float(np.mean(exit_elapsed_pct)) if exit_elapsed_pct else float("nan")
    )
    _targets = df["exit_target"].to_numpy(dtype=float)
    peak_rate = float(np.mean(_targets <= 1e-10))

    metrics = {
        "n_events": int(n_events),
        "mean_improvement_bp": mean_improvement,
        "pct_events_improved": pct_improved,
        "model_mean_pnl": model_mean,
        "rule_mean_pnl": rule_mean,
        "model_sharpe": model_sharpe,
        "rule_sharpe": rule_sharpe,
        "exit_elapsed_pct_mean": exit_elapsed_pct_mean,
        "peak_rate": peak_rate,
    }

    print("\n[EXIT] summary")
    print(pd.DataFrame([metrics]).to_string(index=False))

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (exclusive)")
    p.add_argument("--outdir", type=str, default="./out_vortexbar")
    p.add_argument("--v_target", type=float, default=1500.0)
    p.add_argument("--horizon", type=int, default=18)
    p.add_argument("--cache_dir", type=str, default="./.cache_vortexbar")
    p.add_argument("--dl_workers", type=int, default=4)
    p.add_argument("--rest_timeout", type=float, default=8.0)
    p.add_argument("--rest_sleep", type=float, default=0.2)
    p.add_argument("--rest_max_retries", type=int, default=5)
    p.add_argument("--status_bar", type=int, default=1, choices=[0, 1])
    p.add_argument("--status_every_sec", type=float, default=2.0)
    p.add_argument("--status_width", type=int, default=120)
    p.add_argument("--max_bars", type=int, default=0)
    p.add_argument(
        "--n_lags",
        type=int,
        default=0,
        help="Number of lag bars to append for LAG_FEATURE_COLS (0 = disabled)",
    )
    p.add_argument(
        "--feat_scale",
        type=int,
        default=18,
        help=(
            "Characteristic timescale (bars) for feature engineering. "
            "Controls retHvb lookback, z-score rolling windows, "
            "volatility windows, and VPIN window. "
            "Default 18. Independent of target horizon."
        ),
    )
    p.add_argument(
        "--pctrank_window",
        type=int,
        default=500,
        help=(
            "Rolling window (bars) for causal percentile rank of regime-sensitive "
            "features (ret1vb_bp, retHvb_bp, range1vb_bp, cur_vol_vb_bp, "
            "kyle_lambda, roll_spread, vpin_delta). "
            "Default 500 matches rt3_spread_window default. "
            "Larger = more stable rank, less responsive to regime shifts."
        ),
    )
    # ── Target mode ──────────────────────────────────────────────────────
    p.add_argument(
        "--target_mode",
        type=str,
        default="vtsr",
        choices=["vtsr", "rt3_event", "continuous_alpha"],
        help=(
            "vtsr: fixed-horizon VTSR target (original). "
            "rt3_event: market-determined duration target via RT3 trigger. "
            "continuous_alpha: alpha gating + entry regression pipeline."
        ),
    )
    # ── Continuous Alpha parameters ─────────────────────────────────
    p.add_argument(
        "--ca_horizon",
        type=int,
        default=5,
        help=(
            "Target horizon (bars) for continuous_alpha mode. "
            "Used by both Alpha and Entry models. "
            "Alpha: ER and net_move computed over this horizon. "
            "Entry: forward VWAP return over this horizon. "
            "Default 5. Also determines fixed-H exit."
        ),
    )
    p.add_argument(
        "--ca_alpha_top_pct",
        type=float,
        default=0.20,
        help=(
            "Fraction of bars to select as 'alpha present'. "
            "0.20 = top 20%% by alpha score. "
            "Applied per-fold: training uses actual alpha_target percentile, "
            "OOS uses predicted alpha_score percentile."
        ),
    )
    p.add_argument(
        "--ca_skip_alpha_oos",
        type=int,
        default=0,
        help=(
            "If 1, skip Alpha model OOS filtering: Entry model is evaluated "
            "on ALL OOS bars (not just alpha-passed bars). "
            "Alpha model still runs (for logging + _alpha_feat), and "
            "Entry training undersampling (ER-based) is unchanged. "
            "Default 0 (normal alpha filtering)."
        ),
    )
    # ── RT3 trigger parameters ──────────────────────────────────────────
    p.add_argument(
        "--rt3_spread_window",
        type=int,
        default=500,
        help=(
            "Rolling window (bars) for causal percentile rank of roll_spread. "
            "~1 trading day at v_target=1000. Larger = more stable rank, "
            "less responsive to intraday regime shifts."
        ),
    )
    p.add_argument(
        "--rt3_spread_pct_on",
        type=float,
        default=0.80,
        help=(
            "Percentile threshold for spread shock ON. "
            "0.80 = current spread in top 20%% of recent window. "
            "Raise to require stronger spread expansion (fewer triggers)."
        ),
    )
    p.add_argument(
        "--rt3_spread_pct_off",
        type=float,
        default=0.40,
        help=(
            "Percentile threshold for spread normalization (exit condition A). "
            "0.40 = spread returned to below-median. "
            "Raise to exit earlier; lower to require fuller normalization."
        ),
    )
    p.add_argument(
        "--rt3_k",
        type=int,
        default=4,
        help=(
            "Confirmation window (bars) for spread shock recency and "
            "taker_imb directional consistency. "
            "3-5 recommended. Higher K requires longer confirmation."
        ),
    )
    p.add_argument(
        "--rt3_imb_consistency_on",
        type=float,
        default=0.75,
        help=(
            "taker_imb directional consistency threshold for RT3 ON. "
            "With K=4: 0.75 means 3/4 bars same direction. "
            "Fixed by theory (Glosten-Milgrom); not regime-dependent."
        ),
    )
    p.add_argument(
        "--rt3_imb_consistency_off",
        type=float,
        default=0.50,
        help=(
            "taker_imb directional consistency threshold for exit condition B. "
            "0.50 = directional signal no stronger than random. "
            "Fixed by theory; lower = earlier exit on direction break."
        ),
    )
    p.add_argument(
        "--rt3_hmax",
        type=int,
        default=50,
        help=(
            "Maximum holding bars (safety cap). "
            "If no RT3 exit found within hmax bars, label is NaN. "
            "At v_target=1000, 50 bars ≈ 3-4 hours. "
            "Also determines WF purge_bars when target_mode=rt3_event."
        ),
    )
    p.add_argument(
        "--rt3_off_drop_ratio",
        type=float,
        default=0.30,
        help=(
            "Peak-relative drop ratio that triggers rt3_off. "
            "spread_pct must fall below peak_spread*(1-drop_ratio). "
            "0.30 = 30%% drop from event peak required. "
            "Lower = earlier exit; higher = requires fuller normalization."
        ),
    )
    p.add_argument(
        "--rt3_off_peak_window",
        type=int,
        default=20,
        help=(
            "Bars to look back when computing event-peak spread_pct. "
            "Causal rolling max over this window. "
            "20 bars approx covers the typical event buildup period."
        ),
    )
    p.add_argument(
        "--rt3_off_confirm_bars",
        type=int,
        default=2,
        help=(
            "Consecutive bars the exit_raw signal must hold before "
            "rt3_off fires. 2 = requires 2 consecutive bars below "
            "peak*(1-drop_ratio), preventing single-bar noise exits."
        ),
    )
    # ── Exit model (Structure B) ─────────────────────────────────────────────
    p.add_argument(
        "--exit_n_samples",
        type=int,
        default=300,
        help="Uniform samples per event tick path.",
    )
    p.add_argument(
        "--exit_lag_interval_sec",
        type=float,
        default=5.0,
        help="Lag spacing in seconds for exit features.",
    )
    p.add_argument(
        "--exit_n_lags",
        type=int,
        default=12,
        help="Number of lag steps for exit features.",
    )
    p.add_argument(
        "--exit_threshold",
        type=float,
        default=0.0,
        help=(
            "Alpha threshold for continuation value exit decision. "
            "Exit when predicted_remaining_upside <= alpha. "
            "0.0 = exit when model predicts no further upside (theoretical optimal). "
            "Positive = exit earlier (conservative profit taking). "
            "Negative = wait longer (aggressive, requires clear peak signal)."
        ),
    )
    p.add_argument(
        "--exit_recent_window",
        type=int,
        default=20,
        help="Rolling window size for tick-gap and direction consistency features.",
    )
    p.add_argument(
        "--exit_valid_frac",
        type=float,
        default=0.2,
        help="Validation fraction for exit model early stopping.",
    )
    p.add_argument("--exit_lgbm_n_estimators", type=int, default=2000)
    p.add_argument("--exit_lgbm_num_leaves", type=int, default=31)
    p.add_argument("--exit_lgbm_max_depth", type=int, default=-1)
    p.add_argument("--exit_lgbm_learning_rate", type=float, default=0.05)
    p.add_argument("--exit_lgbm_min_data_in_leaf", type=int, default=50)
    p.add_argument("--exit_lgbm_feature_fraction", type=float, default=0.7)
    p.add_argument("--exit_lgbm_bagging_fraction", type=float, default=0.8)
    p.add_argument("--exit_lgbm_bagging_freq", type=int, default=5)
    p.add_argument("--exit_lgbm_early_stopping_rounds", type=int, default=50)
    p.add_argument(
        "--exit_wf_expanding",
        action="store_true",
        default=True,
        help="Use expanding walk-forward for exit model (default: True).",
    )
    p.add_argument(
        "--no_exit_wf_expanding",
        dest="exit_wf_expanding",
        action="store_false",
        help="Use rolling WF for exit model instead of expanding.",
    )

    # ?? Gate model (confidence-based event filtering) ?????????????????????
    p.add_argument(
        "--gate_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Enable gate model for confidence-based bar filtering. "
            "0=disabled, 1=enabled."
        ),
    )
    p.add_argument(
        "--gate_min_train_folds",
        type=int,
        default=2,
        help=(
            "Minimum number of prior OOS folds required before gate can train. "
            "Folds before this use no gate filtering. Default 2."
        ),
    )
    p.add_argument(
        "--gate_threshold_pct",
        type=float,
        default=70.0,
        help=(
            "Percentile threshold for gate filtering. 70.0 = keep top 30%% of gate scores. "
            "Applied per-fold on OOS gate predictions. Range [0, 100]."
        ),
    )
    p.add_argument(
        "--gate_em_rounds",
        type=int,
        default=0,
        help=(
            "Number of EM re-training rounds. 0 = gate filtering only (no entry re-train). "
            "1 = one round of re-training entry model on gate-filtered data. "
            "Each round: gate filter ? re-train entry ? new preds ? new gate ? ..."
        ),
    )

    p.add_argument("--gate_lgbm_n_estimators", type=int, default=1500)
    p.add_argument("--gate_lgbm_num_leaves", type=int, default=31)
    p.add_argument("--gate_lgbm_max_depth", type=int, default=5)
    p.add_argument("--gate_lgbm_learning_rate", type=float, default=0.03)
    p.add_argument("--gate_lgbm_min_data_in_leaf", type=int, default=150)
    p.add_argument("--gate_lgbm_feature_fraction", type=float, default=0.6)
    p.add_argument("--gate_lgbm_bagging_fraction", type=float, default=0.7)
    p.add_argument("--gate_lgbm_bagging_freq", type=int, default=1)
    p.add_argument("--gate_lgbm_lambda_l2", type=float, default=10.0)
    p.add_argument("--gate_lgbm_early_stopping_rounds", type=int, default=50)
    p.add_argument("--gate_lgbm_valid_frac", type=float, default=0.15)
    p.add_argument("--vol_ewma_span", type=int, default=50, help="EWMA span for volatility in VTSR target (bars)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dir_lgbm_boosting", type=str, default="dart", choices=["dart", "gbdt"])
    p.add_argument("--dir_lgbm_objective", type=str, default="regression")
    p.add_argument("--dir_lgbm_learning_rate", type=float, default=0.03)
    p.add_argument("--dir_lgbm_n_estimators", type=int, default=2000)
    p.add_argument("--dir_lgbm_num_leaves", type=int, default=31)
    p.add_argument("--dir_lgbm_max_depth", type=int, default=-1)
    p.add_argument("--dir_lgbm_min_data_in_leaf", type=int, default=200)
    p.add_argument("--dir_lgbm_min_sum_hessian_in_leaf", type=float, default=1e-3)
    p.add_argument("--dir_lgbm_feature_fraction", type=float, default=0.7)
    p.add_argument("--dir_lgbm_bagging_fraction", type=float, default=0.7)
    p.add_argument("--dir_lgbm_bagging_freq", type=int, default=1)
    p.add_argument("--dir_lgbm_lambda_l1", type=float, default=0.0)
    p.add_argument("--dir_lgbm_lambda_l2", type=float, default=5.0)
    p.add_argument("--dir_lgbm_min_gain_to_split", type=float, default=0.0)
    p.add_argument("--dir_lgbm_max_bin", type=int, default=255)
    p.add_argument("--dir_lgbm_drop_rate", type=float, default=0.1)
    p.add_argument("--dir_lgbm_skip_drop", type=float, default=0.5)
    p.add_argument("--dir_lgbm_max_drop", type=int, default=50)
    p.add_argument("--dir_lgbm_uniform_drop", type=int, default=0)
    p.add_argument("--dir_lgbm_n_jobs", type=int, default=-1)
    p.add_argument("--dir_lgbm_early_stopping_rounds", type=int, default=0)
    p.add_argument("--dir_lgbm_valid_frac", type=float, default=0.15)
    p.add_argument("--dir_lgbm_valid_min", type=int, default=300)
    p.add_argument("--dir_lgbm_verbose", type=int, default=-1)
    p.add_argument("--wf_mode", type=str, default="bars", choices=["bars", "days"])
    p.add_argument("--wf_train_bars", type=int, default=6000)
    p.add_argument("--wf_oos_bars", type=int, default=400)
    p.add_argument("--wf_step_bars", type=int, default=0)
    p.add_argument("--wf_gap_bars", type=int, default=0)
    p.add_argument("--wf_train_days", type=int, default=30)
    p.add_argument("--wf_oos_days", type=int, default=7)
    p.add_argument("--wf_step_days", type=int, default=7)
    p.add_argument("--wf_gap_days", type=int, default=0)
    p.add_argument("--wf_purge_bars", type=int, default=-1)
    p.add_argument(
        "--wf_workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for WF fold execution. "
            "1 = serial. >1 forces lgbm n_jobs=1 per worker to avoid "
            "thread oversubscription."
        ),
    )
    p.add_argument(
        "--wf_expanding",
        action="store_true",
        default=False,
        help=(
            "Use expanding walk-forward window instead of rolling. "
            "When set, each fold trains on ALL data from the start of "
            "the dataset up to tr_end (ignoring wf_train_bars / "
            "wf_train_days as a lower bound). Combines well with "
            "--wf_sample_decay to down-weight distant history."
        ),
    )
    p.add_argument(
        "--wf_sample_decay",
        type=float,
        default=0.0,
        help=(
            "Per-bar exponential decay factor for LightGBM sample weights. "
            "0.0 = disabled (uniform weights, default). "
            "Example: 0.995 -> at a 1000-bar window the oldest bar has "
            "weight 0.995^999 approx 0.007 vs newest bar weight 1.0."
        ),
    )
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if lgb is None:
        raise RuntimeError("lightgbm not installed; pip install lightgbm")
    if str(args.dir_lgbm_objective).lower() != "regression":
        raise ValueError("--dir_lgbm_objective must be 'regression'")
    if not (0.0 < args.dir_lgbm_feature_fraction <= 1.0):
        raise ValueError("--dir_lgbm_feature_fraction must be in (0,1]")
    if not (0.0 < args.dir_lgbm_bagging_fraction <= 1.0):
        raise ValueError("--dir_lgbm_bagging_fraction must be in (0,1]")
    if not (0.0 < args.dir_lgbm_valid_frac <= 1.0):
        raise ValueError("--dir_lgbm_valid_frac must be in (0,1]")
    if args.dir_lgbm_num_leaves < 2:
        raise ValueError("--dir_lgbm_num_leaves must be >= 2")
    if args.dir_lgbm_min_data_in_leaf < 1:
        raise ValueError("--dir_lgbm_min_data_in_leaf must be >= 1")
    if args.dir_lgbm_early_stopping_rounds < 0:
        raise ValueError("--dir_lgbm_early_stopping_rounds must be >= 0")
    if args.wf_purge_bars < -1:
        raise ValueError("--wf_purge_bars must be >= -1")
    if args.n_lags < 0:
        raise ValueError("--n_lags must be >= 0")
    if args.feat_scale < 5:
        raise ValueError("--feat_scale must be >= 5")
    if args.pctrank_window < 20:
        raise ValueError("--pctrank_window must be >= 20")
    if args.wf_workers < 1:
        raise ValueError("--wf_workers must be >= 1")
    if not (0.0 <= args.wf_sample_decay < 1.0):
        raise ValueError("--wf_sample_decay must be in [0, 1)")
    if args.target_mode == "continuous_alpha":
        if args.ca_horizon < 1:
            raise ValueError("--ca_horizon must be >= 1")
        if not (0.0 < args.ca_alpha_top_pct <= 1.0):
            raise ValueError("--ca_alpha_top_pct must be in (0, 1]")
    if args.target_mode == "rt3_event":
        if not (0.0 < args.rt3_spread_pct_on <= 1.0):
            raise ValueError("--rt3_spread_pct_on must be in (0, 1]")
        if not (0.0 <= args.rt3_spread_pct_off < args.rt3_spread_pct_on):
            raise ValueError(
                "--rt3_spread_pct_off must be in [0, rt3_spread_pct_on)"
            )
        if args.rt3_k < 2:
            raise ValueError("--rt3_k must be >= 2")
        if not (0.0 < args.rt3_imb_consistency_on <= 1.0):
            raise ValueError("--rt3_imb_consistency_on must be in (0, 1]")
        if not (0.0 <= args.rt3_imb_consistency_off < args.rt3_imb_consistency_on):
            raise ValueError(
                "--rt3_imb_consistency_off must be in [0, rt3_imb_consistency_on)"
            )
        if args.rt3_hmax < 2:
            raise ValueError("--rt3_hmax must be >= 2")
        if args.rt3_spread_window < 10:
            raise ValueError("--rt3_spread_window must be >= 10")
        if not (0.0 < args.rt3_off_drop_ratio < 1.0):
            raise ValueError("--rt3_off_drop_ratio must be in (0, 1)")
        if args.rt3_off_peak_window < 2:
            raise ValueError("--rt3_off_peak_window must be >= 2")
        if args.rt3_off_confirm_bars < 1:
            raise ValueError("--rt3_off_confirm_bars must be >= 1")
    run(args)
    print("Example:")
    print(
        "python vortexbar_lab.py --symbol BTCUSDT --start 2024-01-01 --end 2024-02-15 "
        "--outdir ./out_vortexbar --v_target 1500 --horizon 18 --vol_ewma_span 50"
    )
