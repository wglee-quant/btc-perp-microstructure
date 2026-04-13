"""
ob_poc_v4.py
Time-grid walk-forward test for Bybit BTCUSDT with variable horizon and domain-score filtering.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import io
import json
import math
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import numba as nb
except Exception:
    nb = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

EPS = 1e-12

LAG_FEATURE_COLS: list[str] = [
    "taker_imb1vb",
    "par",
    "twi",
    "large_imb_share",
    "vpin_delta",
    "ret1vb_bp",
    "range1vb_bp",
    "pde",
    "close_pos1vb",
    "vol_ratio",
    "dt_sec_z",
    "kyle_lambda",
    "roll_spread",
    "ret1vb_bp_pctrank",
    "range1vb_bp_pctrank",
    "kyle_lambda_pctrank",
    "roll_spread_pctrank",
    "vpin_delta_pctrank",
]

OB_RAW_FEATURES = [
    "ob_imb_l1",
    "ob_imb_l5",
    "ob_imb_l10",
    "ob_depth_ratio_5",
    "ob_depth_ratio_10",
    "ob_spread",
    "ob_spread_rel",
    "ob_microprice_adj",
    "ob_total_depth_5",
]

OB_PCTRANK_COLS = [
    "ob_spread_close",
    "ob_spread_rel_close",
    "ob_total_depth_5_close",
    "ob_microprice_adj_close",
]

OB_LAG_CLOSE_COLS = [
    "ob_imb_l1_close",
    "ob_imb_l5_close",
    "ob_imb_l10_close",
    "ob_depth_ratio_5_close",
    "ob_depth_ratio_10_close",
    "ob_spread_close",
    "ob_spread_rel_close",
    "ob_microprice_adj_close",
    "ob_total_depth_5_close",
]

LAG_FEATURE_COLS_V3: list[str] = [
    "taker_imb",
    "pde",
    "kyle_lambda",
    "roll_spread",
    "ret1_bp",
    "range_bp",
    "close_pos",
    "close_vs_vwap",
    "par",
    "twi",
    "large_imb_share",
    "entropy_norm",
    "burst_cv",
    "rev_count_ratio",
    "dt_sec",
    "trade_count",
]

PCTRANK_COLS_V3: list[str] = [
    "ret1_bp",
    "retH_bp",
    "range_bp",
    "cur_vol_bp",
    "kyle_lambda",
    "roll_spread",
    "vpin_delta",
]

OB_LAG_COLS_V3: list[str] = [
    "ob_imb_l1_close",
    "ob_imb_l5_close",
    "ob_imb_l10_close",
    "ob_depth_ratio_5_close",
    "ob_depth_ratio_10_close",
    "ob_spread_close",
    "ob_spread_rel_close",
    "ob_microprice_adj_close",
    "ob_total_depth_5_close",
]

BYBIT_API_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}

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

def safe_div_scalar(a: float, b: float, eps: float = EPS) -> float:
    return float(a) / (float(b) + float(eps))


def build_bars_at_grid(
    ts_all: np.ndarray,
    price_all: np.ndarray,
    qty_all: np.ndarray,
    ibm_all: np.ndarray,
    T_ms: int,
    v_target: float,
    n_bars: int,
    qty_cumsum: np.ndarray | None = None,
) -> pd.DataFrame | None:
    end_idx = int(np.searchsorted(ts_all, int(T_ms), side="right"))
    if end_idx <= 0:
        return None

    if qty_cumsum is None:
        qty_cumsum = np.cumsum(qty_all, dtype=np.float64)
    total_qty = float(qty_cumsum[end_idx - 1])
    need_qty = float(v_target) * float(n_bars)
    if total_qty + EPS < need_qty:
        return None

    cutoff_qty = total_qty - need_qty
    if cutoff_qty <= 0.0:
        start_idx = 0
    else:
        start_idx = int(np.searchsorted(qty_cumsum[:end_idx], cutoff_qty, side="right"))

    df_bars = build_volume_bars(
        ts_all[start_idx:end_idx],
        price_all[start_idx:end_idx],
        qty_all[start_idx:end_idx],
        ibm_all[start_idx:end_idx],
        v_target,
    )

    if len(df_bars) < n_bars:
        return None
    return df_bars.iloc[-n_bars:].reset_index(drop=True)


def query_ob_snapshot(
    ob_ts: np.ndarray,
    raw_feats: Dict[str, np.ndarray],
    mid_prices: np.ndarray,
    target_ts: int,
    tolerance_ms: int = 2000,
) -> tuple[Dict[str, float], float]:
    j = int(np.searchsorted(ob_ts, int(target_ts), side="right")) - 1
    if j < 0 or j >= len(ob_ts) or abs(int(ob_ts[j]) - int(target_ts)) > int(tolerance_ms):
        return {feat: np.nan for feat in OB_RAW_FEATURES}, np.nan
    return {feat: float(raw_feats[feat][j]) for feat in OB_RAW_FEATURES}, float(mid_prices[j])


def attach_ob_features(
    bars_df: pd.DataFrame,
    ob_ts: np.ndarray,
    raw_feats: Dict[str, np.ndarray],
    mid_prices: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for bar_idx in range(len(bars_df)):
        bar = bars_df.iloc[bar_idx]
        snapshot, mid = query_ob_snapshot(ob_ts, raw_feats, mid_prices, int(bar["ts_ms"]))
        merged = dict(snapshot)
        merged["mid_price"] = mid
        out[bar_idx] = merged
    return out


def _ob_row_to_features(row: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "ob_imb_l1_close": float(row.get("ob_imb_l1", np.nan)),
        "ob_imb_l5_close": float(row.get("ob_imb_l5", np.nan)),
        "ob_imb_l10_close": float(row.get("ob_imb_l10", np.nan)),
        "ob_depth_ratio_5_close": float(row.get("ob_depth_ratio_5", np.nan)),
        "ob_depth_ratio_10_close": float(row.get("ob_depth_ratio_10", np.nan)),
        "ob_spread_rel_close": float(row.get("ob_spread_rel", np.nan)),
        "ob_microprice_adj_close": float(row.get("ob_microprice_adj", np.nan)),
        "ob_total_depth_5_close": float(row.get("ob_total_depth_5", np.nan)),
    }
    out["ob_spread_close"] = float(row.get("ob_spread", np.nan))
    out["ob_mid_price"] = float(row.get("mid_price", np.nan))
    return out


def format_ob_features(ob_feat: Dict[int, Dict[str, float]], n_lags: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    k_cur = len(ob_feat) - 1
    cur = _ob_row_to_features(ob_feat.get(k_cur, {}))
    out.update(cur)

    for lag in range(1, n_lags + 1):
        idx = k_cur - lag
        lag_row = _ob_row_to_features(ob_feat.get(idx, {})) if idx >= 0 else {}
        for col in OB_LAG_COLS_V3:
            out[f"{col}_lag{lag}"] = float(lag_row.get(col, np.nan))
    return out


def compute_grid_base_features(df_bars: pd.DataFrame, n_lags: int) -> Dict[str, float]:
    K = len(df_bars) - 1
    row = df_bars.iloc[K]

    feat: Dict[str, float] = {
        "taker_imb": safe_div_scalar(
            row["taker_buy_vol"] - row["taker_sell_vol"],
            row["taker_buy_vol"] + row["taker_sell_vol"],
        ),
        "pde": float(row["pde"]),
        "range_bp": float(row["range1vb_bp"]),
        "par": float(row["par"]),
        "twi": float(row["twi"]),
        "large_imb_share": float(row["large_imb_share"]),
        "entropy_norm": float(row["entropy_norm"]),
        "burst_cv": float(row["burst_cv"]),
        "rev_count_ratio": float(row["rev_count_ratio"]),
        "kyle_lambda": float(row["kyle_lambda"]),
        "roll_spread": float(row["roll_spread"]),
        "dt_sec": float(row["dt_sec"]),
        "trade_count": float(row["trade_count"]),
        "close": float(row["close"]),
        "vwap": float(row["vwap"]),
        "volume": float(row["volume"]),
        "taker_buy_vol": float(row["taker_buy_vol"]),
        "taker_sell_vol": float(row["taker_sell_vol"]),
        "mid_size_share": float(row["mid_size_share"]),
    }

    hl = float(row["high"] - row["low"])
    feat["close_pos"] = ((float(row["close"]) - float(row["low"])) / hl - 0.5) if abs(hl) >= EPS else np.nan

    if K >= 1:
        prev_close = float(df_bars.iloc[K - 1]["close"])
        cur_close = float(row["close"])
        feat["ret1_bp"] = safe_div_scalar(cur_close - prev_close, prev_close) * 1e4 if prev_close > EPS else np.nan
        feat["close_vs_vwap"] = (cur_close - float(row["vwap"])) / max(hl, EPS)
    else:
        feat["ret1_bp"] = np.nan
        feat["close_vs_vwap"] = np.nan

    for lag in range(1, n_lags + 1):
        idx = K - lag
        if idx < 0:
            for col in LAG_FEATURE_COLS_V3:
                feat[f"{col}_lag{lag}"] = np.nan
            continue

        r = df_bars.iloc[idx]
        hl_l = float(r["high"] - r["low"])
        taker_imb_l = safe_div_scalar(
            r["taker_buy_vol"] - r["taker_sell_vol"],
            r["taker_buy_vol"] + r["taker_sell_vol"],
        )
        if idx >= 1:
            prev_c = float(df_bars.iloc[idx - 1]["close"])
            ret_l = safe_div_scalar(float(r["close"]) - prev_c, prev_c) * 1e4 if prev_c > EPS else np.nan
        else:
            ret_l = np.nan

        lag_vals: Dict[str, float] = {
            "taker_imb": taker_imb_l,
            "pde": float(r["pde"]),
            "kyle_lambda": float(r["kyle_lambda"]),
            "roll_spread": float(r["roll_spread"]),
            "ret1_bp": ret_l,
            "range_bp": float(r["range1vb_bp"]),
            "close_pos": ((float(r["close"]) - float(r["low"])) / hl_l - 0.5) if abs(hl_l) >= EPS else np.nan,
            "close_vs_vwap": (float(r["close"]) - float(r["vwap"])) / max(hl_l, EPS),
            "par": float(r["par"]),
            "twi": float(r["twi"]),
            "large_imb_share": float(r["large_imb_share"]),
            "entropy_norm": float(r["entropy_norm"]),
            "burst_cv": float(r["burst_cv"]),
            "rev_count_ratio": float(r["rev_count_ratio"]),
            "dt_sec": float(r["dt_sec"]),
            "trade_count": float(r["trade_count"]),
        }
        for col in LAG_FEATURE_COLS_V3:
            feat[f"{col}_lag{lag}"] = lag_vals[col]

    return feat


def compute_rolling_features(
    df_grid: pd.DataFrame,
    feat_scale: int,
    pctrank_window: int,
) -> pd.DataFrame:
    df_grid = df_grid.sort_values("ts_ms").reset_index(drop=True)

    window_v = max(10, int(4 * feat_scale))
    minp_v = max(5, window_v // 4)
    df_grid["dt_sec_z"] = robust_rolling_z(
        np.log1p(df_grid["dt_sec"].clip(lower=0.0)),
        window=window_v,
        min_periods=minp_v,
    )
    df_grid["trade_count_z"] = robust_rolling_z(
        np.log1p(df_grid["trade_count"].clip(lower=0.0)),
        window=window_v,
        min_periods=minp_v,
    )

    tc_log = np.log(df_grid["trade_count"].clip(lower=EPS))
    tc_mean = (
        df_grid["trade_count"]
        .rolling(feat_scale, min_periods=max(3, feat_scale // 4))
        .mean()
        .clip(lower=EPS)
    )
    df_grid["tca"] = tc_log - np.log(tc_mean)

    ret_bp = df_grid["ret1_bp"]
    vol_win = max(10, feat_scale)
    minp = max(3, vol_win // 4)
    vol_ret = np.sqrt((ret_bp * ret_bp).rolling(vol_win, min_periods=minp).mean())
    range_mean = df_grid["range_bp"].rolling(vol_win, min_periods=minp).mean()
    df_grid["cur_vol_bp"] = np.maximum(vol_ret, range_mean)

    vol_short_N = max(5, feat_scale // 2)
    alpha_s = 2.0 / (vol_short_N + 1.0)
    alpha_l = 2.0 / (vol_win + 1.0)
    vol_short = ret_bp.abs().ewm(alpha=alpha_s, adjust=False).mean()
    vol_long = ret_bp.abs().ewm(alpha=alpha_l, adjust=False).mean()
    df_grid["vol_ratio"] = safe_div(vol_short, vol_long + EPS)

    df_grid["retH_bp"] = safe_div(
        df_grid["close"] - df_grid["close"].shift(feat_scale),
        df_grid["close"].shift(feat_scale),
    ) * 1e4

    imb_abs = (df_grid["taker_buy_vol"] - df_grid["taker_sell_vol"]).abs()
    N_vpin = max(10, feat_scale)
    vpin_num = imb_abs.rolling(N_vpin, min_periods=max(3, N_vpin // 4)).sum()
    vpin_den = df_grid["volume"].rolling(N_vpin, min_periods=max(3, N_vpin // 4)).sum()
    df_grid["vpin_N"] = safe_div(vpin_num, vpin_den)
    df_grid["vpin_delta"] = df_grid["vpin_N"].diff(1)

    _pctrank_minp = max(10, pctrank_window // 10)
    for col in PCTRANK_COLS_V3:
        if col in df_grid.columns:
            df_grid[f"{col}_pctrank"] = (
                df_grid[col]
                .rolling(pctrank_window, min_periods=_pctrank_minp)
                .rank(pct=True)
            )
    return df_grid


def compute_time_based_target_vec(
    trade_ts: np.ndarray,
    trade_price: np.ndarray,
    grid_times_ms: np.ndarray,
    target_horizon_ms: int,
) -> np.ndarray:
    j_now = np.searchsorted(trade_ts, grid_times_ms, side="right") - 1
    j_future = np.searchsorted(trade_ts, grid_times_ms + int(target_horizon_ms), side="right") - 1

    valid = (j_now >= 0) & (j_future >= 0) & (j_future < len(trade_ts))
    j_now_safe = np.clip(j_now, 0, max(len(trade_ts) - 1, 0))
    j_future_safe = np.clip(j_future, 0, max(len(trade_ts) - 1, 0))

    p_now = trade_price[j_now_safe]
    p_future = trade_price[j_future_safe]

    gap = np.abs(trade_ts[j_future_safe] - (grid_times_ms + int(target_horizon_ms)))
    valid &= gap < 5000

    y = np.full(len(grid_times_ms), np.nan, dtype=np.float64)
    mask = valid & (p_now > EPS) & (p_future > EPS)
    y[mask] = np.log(p_future[mask] / p_now[mask]) * 10000.0
    return y


def compute_variable_target_vec(
    grid_close: np.ndarray,
    grid_taker_imb: np.ndarray,
    max_hold_bars: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised variable-horizon target.
    For each grid point, find the first future bar where taker_imb sign flips.
    If no flip within max_hold_bars, use max_hold_bars close.
    """
    N = len(grid_close)
    y = np.full(N, np.nan, dtype=np.float64)
    hold_bars = np.full(N, np.nan, dtype=np.float64)
    entry_sign = np.sign(grid_taker_imb)
    valid_entry = np.isfinite(entry_sign) & (entry_sign != 0) & (grid_close > 0)
    active = valid_entry.copy()

    max_hold = int(max_hold_bars)
    for j in range(1, max_hold + 1):
        future_sign = np.empty(N, dtype=np.float64)
        future_sign[:N - j] = np.sign(grid_taker_imb[j:])
        future_sign[N - j:] = 0.0

        future_close = np.empty(N, dtype=np.float64)
        future_close[:N - j] = grid_close[j:]
        future_close[N - j:] = np.nan

        flipped_here = active & (future_sign == -entry_sign)
        valid_exit = flipped_here & np.isfinite(future_close) & (future_close > 0)
        y[valid_exit] = np.log(future_close[valid_exit] / grid_close[valid_exit]) * 10000.0
        hold_bars[valid_exit] = float(j)
        active[flipped_here] = False

    still_nan = active & ((np.arange(N) + max_hold) < N)
    if np.any(still_nan):
        idx = np.where(still_nan)[0]
        future_close_max = grid_close[idx + max_hold]
        ok = future_close_max > 0
        y[idx[ok]] = np.log(future_close_max[ok] / grid_close[idx[ok]]) * 10000.0
        hold_bars[idx[ok]] = float(max_hold)

    return y, hold_bars


def build_trade_feature_cols(n_lags: int = 0) -> List[str]:
    cols = [
        "ret1_bp",
        "retH_bp",
        "range_bp",
        "close_pos",
        "close_vs_vwap",
        "pde",
        "cur_vol_bp",
        "vol_ratio",
        "taker_imb",
        "par",
        "twi",
        "large_imb_share",
        "vpin_N",
        "vpin_delta",
        "rev_count_ratio",
        "entropy_norm",
        "kyle_lambda",
        "roll_spread",
        "dt_sec_z",
        "trade_count_z",
        "tca",
        "burst_cv",
    ]
    for col in PCTRANK_COLS_V3:
        cols.append(f"{col}_pctrank")
    for col in LAG_FEATURE_COLS_V3:
        for lag in range(1, n_lags + 1):
            cols.append(f"{col}_lag{lag}")
    return cols


def build_v4_feature_cols(n_lags: int = 2) -> list[str]:
    """v4 feature set with reduced lag count."""
    trade_base = [
        "ret1_bp", "retH_bp", "range_bp", "close_pos", "close_vs_vwap",
        "taker_imb", "pde", "kyle_lambda", "roll_spread",
        "large_imb_share", "entropy_norm", "burst_cv",
        "trade_count", "mid_size_share", "rev_count_ratio",
        "par", "twi", "vpin_delta", "tca",
        "dt_sec", "dt_sec_z", "trade_count_z",
    ]
    trade_lag_base = [
        "taker_imb", "pde", "kyle_lambda", "roll_spread",
        "ret1_bp", "range_bp", "close_pos", "close_vs_vwap",
        "large_imb_share", "entropy_norm", "burst_cv",
        "trade_count", "mid_size_share", "rev_count_ratio",
        "par", "twi", "vpin_delta", "tca",
    ]
    rolling = [
        "cur_vol_bp", "vol_ratio", "vpin_N",
        "ret1_bp_pctrank", "retH_bp_pctrank", "range_bp_pctrank",
        "cur_vol_bp_pctrank", "kyle_lambda_pctrank",
        "roll_spread_pctrank", "vpin_delta_pctrank",
    ]
    ob_base = [
        "ob_imb_l1_close", "ob_imb_l5_close", "ob_imb_l10_close",
        "ob_depth_ratio_5_close", "ob_depth_ratio_10_close",
        "ob_spread_close", "ob_spread_rel_close",
        "ob_microprice_adj_close", "ob_total_depth_5_close",
        "ob_mid_price",
    ]
    ob_pctrank = [
        "ob_spread_close_pctrank", "ob_spread_rel_close_pctrank",
        "ob_total_depth_5_close_pctrank", "ob_microprice_adj_close_pctrank",
    ]
    ob_lag_base = [
        "ob_imb_l1_close", "ob_imb_l5_close", "ob_imb_l10_close",
        "ob_depth_ratio_5_close", "ob_depth_ratio_10_close",
        "ob_spread_close", "ob_spread_rel_close",
        "ob_microprice_adj_close", "ob_total_depth_5_close",
    ]
    cols = trade_base + rolling + ob_base + ob_pctrank
    for base in trade_lag_base:
        for lag in range(1, n_lags + 1):
            cols.append(f"{base}_lag{lag}")
    for base in ob_lag_base:
        for lag in range(1, n_lags + 1):
            cols.append(f"{base}_lag{lag}")
    return cols


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

def compute_ob_raw_features_vectorized(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:

    bid_sum5 = bid_sizes[:, :5].sum(axis=1)
    ask_sum5 = ask_sizes[:, :5].sum(axis=1)
    bid_sum10 = bid_sizes[:, :10].sum(axis=1)
    ask_sum10 = ask_sizes[:, :10].sum(axis=1)

    mid = (bid_prices[:, 0] + ask_prices[:, 0]) / 2.0

    features = {}
    features["ob_imb_l1"] = safe_div(bid_sizes[:, 0] - ask_sizes[:, 0], bid_sizes[:, 0] + ask_sizes[:, 0] + EPS)
    features["ob_imb_l5"] = safe_div(bid_sum5 - ask_sum5, bid_sum5 + ask_sum5 + EPS)
    features["ob_imb_l10"] = safe_div(bid_sum10 - ask_sum10, bid_sum10 + ask_sum10 + EPS)
    features["ob_depth_ratio_5"] = safe_div(bid_sum5, ask_sum5 + EPS)
    features["ob_depth_ratio_10"] = safe_div(bid_sum10, ask_sum10 + EPS)
    spread = ask_prices[:, 0] - bid_prices[:, 0]
    features["ob_spread"] = spread
    features["ob_spread_rel"] = safe_div(spread, mid + EPS)
    micro = safe_div(
        bid_prices[:, 0] * ask_sizes[:, 0] + ask_prices[:, 0] * bid_sizes[:, 0],
        bid_sizes[:, 0] + ask_sizes[:, 0] + EPS,
    ) - mid
    features["ob_microprice_adj"] = micro
    features["ob_total_depth_5"] = bid_sum5 + ask_sum5

    return features, mid


def _range_sum(cum: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    res = cum[end]
    mask = start > 0
    if np.any(mask):
        res[mask] = res[mask] - cum[start[mask] - 1]
    return res

def _align_ob_raw_chunk(
    bar_start_ts: np.ndarray,
    bar_end_ts: np.ndarray,
    ob_ts: np.ndarray,
    raw_feats: Dict[str, np.ndarray],
    mid_prices: np.ndarray,
    out_close: Dict[str, np.ndarray],
    out_mean: Dict[str, np.ndarray],
    out_delta: Dict[str, np.ndarray],
    out_mid: np.ndarray,
) -> int:
    if len(bar_end_ts) == 0 or len(ob_ts) == 0:
        return 0

    chunk_ts_min = ob_ts[0]
    chunk_ts_max = ob_ts[-1]
    bar_mask = (bar_end_ts >= chunk_ts_min) & (bar_end_ts <= chunk_ts_max)
    if not np.any(bar_mask):
        return 0

    bar_indices = np.where(bar_mask)[0]
    start_idx = np.searchsorted(ob_ts, bar_start_ts[bar_indices], side="left")
    end_idx = np.searchsorted(ob_ts, bar_end_ts[bar_indices], side="right") - 1

    n_obs = len(ob_ts)
    valid = (end_idx >= 0) & (start_idx <= end_idx) & (start_idx < n_obs)
    if not np.any(valid):
        return 0

    valid_indices = bar_indices[valid]
    start_v = start_idx[valid]
    end_v = end_idx[valid]

    mid_fill = np.isnan(out_mid[valid_indices])
    if np.any(mid_fill):
        out_mid[valid_indices[mid_fill]] = mid_prices[end_v[mid_fill]]

    for feat in OB_RAW_FEATURES:
        vals = raw_feats[feat]
        finite = np.isfinite(vals).astype(np.int64)
        vals0 = np.where(np.isfinite(vals), vals, 0.0)
        csum = np.cumsum(vals0)
        ccnt = np.cumsum(finite)
        sum_v = _range_sum(csum, start_v, end_v)
        cnt_v = _range_sum(ccnt, start_v, end_v)
        mean_v = np.where(cnt_v > 0, sum_v / cnt_v, np.nan)
        close_v = vals[end_v]
        start_val = vals[start_v]
        delta_v = close_v - start_val

        close_arr = out_close[feat]
        mean_arr = out_mean[feat]
        delta_arr = out_delta[feat]
        fill_mask = np.isnan(close_arr[valid_indices])
        if np.any(fill_mask):
            idx = valid_indices[fill_mask]
            close_arr[idx] = close_v[fill_mask]
            mean_arr[idx] = mean_v[fill_mask]
            delta_arr[idx] = delta_v[fill_mask]

    return int(np.sum(valid))

def _build_ob_feat_df(
    out_close: Dict[str, np.ndarray],
    out_mean: Dict[str, np.ndarray],
    out_delta: Dict[str, np.ndarray],
    out_mid: np.ndarray,
    index: pd.Index,
    n_lags: int,
    pctrank_window: int,
) -> pd.DataFrame:
    out: Dict[str, np.ndarray] = {}
    for feat in OB_RAW_FEATURES:
        out[f"{feat}_close"] = out_close[feat]
        out[f"{feat}_mean"] = out_mean[feat]
        out[f"{feat}_delta"] = out_delta[feat]

    df_out = pd.DataFrame(out, index=index)
    df_out["ob_mid_price"] = out_mid

    _pctrank_window = int(pctrank_window)
    _pctrank_minp = max(10, _pctrank_window // 10)
    for col in OB_PCTRANK_COLS:
        if col in df_out.columns:
            df_out[f"{col}_pctrank"] = (
                df_out[col]
                .rolling(_pctrank_window, min_periods=_pctrank_minp)
                .rank(pct=True)
            )

    if n_lags > 0:
        for col in OB_LAG_CLOSE_COLS:
            for lag in range(1, n_lags + 1):
                df_out[f"{col}_lag{lag}"] = df_out[col].shift(lag)

    return df_out

def align_ob_to_vbars(
    df_vb: pd.DataFrame,
    df_ob: pd.DataFrame,
    n_lags: int,
    pctrank_window: int,
) -> pd.DataFrame:
    n_bars = len(df_vb)
    if n_bars == 0:
        return pd.DataFrame()

    if df_ob is None or df_ob.empty:
        cols = build_ob_feature_cols(n_lags)
        out = pd.DataFrame(np.nan, index=df_vb.index, columns=cols)
        out["ob_mid_price"] = np.nan
        return out

    ob_ts = df_ob["ts_ms"].to_numpy(dtype=np.int64)
    ensure_monotonic(ob_ts, "orderbook ts_ms")
    bar_start = df_vb["ts_first"].to_numpy(dtype=np.int64)
    bar_end = df_vb["ts_ms"].to_numpy(dtype=np.int64)

    bid_prices = df_ob[[f"bid_price_{i}" for i in range(25)]].to_numpy(dtype=float)
    bid_sizes = df_ob[[f"bid_size_{i}" for i in range(25)]].to_numpy(dtype=float)
    ask_prices = df_ob[[f"ask_price_{i}" for i in range(25)]].to_numpy(dtype=float)
    ask_sizes = df_ob[[f"ask_size_{i}" for i in range(25)]].to_numpy(dtype=float)
    raw_feats, mid_prices = compute_ob_raw_features_vectorized(
        bid_prices,
        bid_sizes,
        ask_prices,
        ask_sizes,
    )

    out_close: Dict[str, np.ndarray] = {}
    out_mean: Dict[str, np.ndarray] = {}
    out_delta: Dict[str, np.ndarray] = {}
    for feat in OB_RAW_FEATURES:
        out_close[feat] = np.full(n_bars, np.nan, dtype=float)
        out_mean[feat] = np.full(n_bars, np.nan, dtype=float)
        out_delta[feat] = np.full(n_bars, np.nan, dtype=float)

    out_mid = np.full(n_bars, np.nan, dtype=float)

    _align_ob_raw_chunk(
        bar_start,
        bar_end,
        ob_ts,
        raw_feats,
        mid_prices,
        out_close,
        out_mean,
        out_delta,
        out_mid,
    )

    return _build_ob_feat_df(out_close, out_mean, out_delta, out_mid, df_vb.index, n_lags, pctrank_window)


def build_ob_feature_cols(n_lags: int) -> List[str]:
    cols = [
        "ob_mid_price",
        "ob_imb_l1_close",
        "ob_imb_l5_close",
        "ob_imb_l10_close",
        "ob_depth_ratio_5_close",
        "ob_depth_ratio_10_close",
        "ob_spread_close",
        "ob_spread_rel_close",
        "ob_microprice_adj_close",
        "ob_total_depth_5_close",
    ]
    cols.extend(
        [
            "ob_spread_close_pctrank",
            "ob_spread_rel_close_pctrank",
            "ob_total_depth_5_close_pctrank",
            "ob_microprice_adj_close_pctrank",
        ]
    )
    for col in OB_LAG_COLS_V3:
        for lag in range(1, n_lags + 1):
            cols.append(f"{col}_lag{lag}")
    return cols

def build_ob_feature_cols_v2(n_lags: int) -> List[str]:
    return build_ob_feature_cols(n_lags)

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

def download_bybit_trades(
    symbol: str,
    date: _dt.date,
    cache_dir: str,
    timeout: float,
    max_retries: int,
) -> str:
    dstr = date.strftime("%Y-%m-%d")
    fname = f"{symbol}{dstr}.csv.gz"
    rel = os.path.join("bybit_trades", symbol, fname)
    fpath = os.path.join(cache_dir, rel)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        return fpath

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    url = f"https://public.bybit.com/trading/{symbol}/{symbol}{dstr}.csv.gz"
    last_err = None
    for _ in range(max_retries):
        tmp_path = fpath + ".tmp"
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, fpath)
            return fpath
        except Exception as e:
            last_err = e
            time.sleep(1.0)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    raise RuntimeError(f"trade download failed: {url} ({last_err})")


def _read_bybit_trade_arrays(
    csv_gz_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz_path = csv_gz_path
    if npz_path.endswith(".csv.gz"):
        npz_path = npz_path[:-7] + ".tick.npz"
    else:
        npz_path = npz_path + ".tick.npz"

    try:
        if os.path.exists(npz_path):
            if os.path.getmtime(npz_path) >= os.path.getmtime(csv_gz_path):
                data = np.load(npz_path)
                return (
                    data["ts_ms"],
                    data["price"],
                    data["qty"],
                    data["ibm"],
                )
    except Exception:
        pass

    try:
        df = pd.read_csv(
            csv_gz_path,
            compression="gzip",
            usecols=["timestamp", "side", "size", "price"],
        )
    except (MemoryError, Exception) as e:
        msg = str(e).lower()
        if "out of memory" not in msg and "memory" not in msg:
            raise
        _chunks_ts = []
        _chunks_price = []
        _chunks_qty = []
        _chunks_ibm = []
        for chunk in pd.read_csv(
            csv_gz_path,
            compression="gzip",
            usecols=["timestamp", "side", "size", "price"],
            chunksize=500_000,
        ):
            _chunks_ts.append(
                (chunk["timestamp"].to_numpy(dtype=np.float64) * 1000).astype(np.int64)
            )
            _chunks_price.append(chunk["price"].to_numpy(dtype=np.float64))
            _chunks_qty.append(chunk["size"].to_numpy(dtype=np.float64))
            _chunks_ibm.append(
                (chunk["side"].astype(str).str.strip().str.lower() == "sell")
                .to_numpy(dtype=bool)
            )
            del chunk

        ts_ms = np.concatenate(_chunks_ts)
        price = np.concatenate(_chunks_price)
        qty = np.concatenate(_chunks_qty)
        ibm = np.concatenate(_chunks_ibm)
        del _chunks_ts, _chunks_price, _chunks_qty, _chunks_ibm

        try:
            np.savez(npz_path, ts_ms=ts_ms, price=price, qty=qty, ibm=ibm)
        except Exception:
            pass

        return ts_ms, price, qty, ibm

    ts_ms = (df["timestamp"].to_numpy(dtype=np.float64) * 1000).astype(np.int64)
    price = df["price"].to_numpy(dtype=np.float64)
    qty = df["size"].to_numpy(dtype=np.float64)
    ibm = (df["side"].astype(str).str.strip().str.lower() == "sell").to_numpy(dtype=bool)

    del df

    try:
        np.savez(npz_path, ts_ms=ts_ms, price=price, qty=qty, ibm=ibm)
    except Exception:
        pass

    return ts_ms, price, qty, ibm


def download_bybit_ob(
    symbol: str,
    start_str: str,
    end_str: str,
    cache_dir: str,
    timeout: float,
    max_retries: int,
    dl_workers: int = 8,
) -> list[str]:
    """Download daily OB ZIPs from quote-saver.bycsi.com."""
    start_date = _dt.datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = _dt.datetime.strptime(end_str, "%Y-%m-%d").date()

    base_url = "https://quote-saver.bycsi.com/orderbook/linear"
    ob_depth_change_date = _dt.date(2025, 8, 21)
    ob_dir = os.path.join(cache_dir, "bybit_ob", symbol)
    os.makedirs(ob_dir, exist_ok=True)

    tasks: list[tuple[str, str]] = []
    cached: list[str] = []
    d = start_date
    while d < end_date:
        date_str = d.strftime("%Y-%m-%d")
        if d >= ob_depth_change_date:
            depth_tag = "ob200"
        else:
            depth_tag = "ob500"
        filename = f"{date_str}_{symbol}_{depth_tag}.data.zip"
        url = f"{base_url}/{symbol}/{filename}"
        fpath = os.path.join(ob_dir, filename)

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            cached.append(fpath)
        else:
            tasks.append((url, fpath))
        d += _dt.timedelta(days=1)

    print(f"[OB] {symbol}: {len(cached)} cached, {len(tasks)} to download")

    def _download_one(item: tuple[str, str]) -> str | None:
        url, fpath = item
        last_err = None
        for attempt in range(max_retries):
            tmp_path = fpath + ".tmp"
            try:
                with requests.get(
                    url, stream=True, timeout=timeout, headers=BYBIT_API_HEADERS
                ) as r:
                    if r.status_code == 404:
                        return None
                    r.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                os.replace(tmp_path, fpath)
                return fpath
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        fname = os.path.basename(fpath)
        print(f"[OB] download failed: {fname} ({last_err})")
        return None

    downloaded: list[str] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=dl_workers) as pool:
            results = pool.map(_download_one, tasks)
            for fpath in results:
                if fpath is not None:
                    downloaded.append(fpath)

    all_paths = sorted(set(cached + downloaded))
    print(f"[OB] Total OB files: {len(all_paths)}")
    return all_paths


def _load_ob_day_arrays(
    zip_path: str,
    ob_interval_ms: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct full orderbook state from snapshot + delta messages.
    Sample the reconstructed book every ob_interval_ms milliseconds.
    Returns top 25 levels for bid/ask prices and sizes.
    """
    if zip_path.endswith(".zip"):
        npz_path = zip_path[:-4] + f".ob_recon_{ob_interval_ms}.npz"
    else:
        npz_path = zip_path + f".ob_recon_{ob_interval_ms}.npz"

    try:
        if os.path.exists(npz_path):
            if os.path.getmtime(npz_path) >= os.path.getmtime(zip_path):
                data = np.load(npz_path)
                return (
                    data["ts_ms"],
                    data["bid_prices"],
                    data["bid_sizes"],
                    data["ask_prices"],
                    data["ask_sizes"],
                )
    except Exception:
        pass

    try:
        import orjson as _orjson
        _loads = _orjson.loads
    except Exception:
        _loads = json.loads

    bid_book: dict[float, float] = {}
    ask_book: dict[float, float] = {}
    book_initialized = False
    last_sample_ts = -(10**18)

    result_ts: list[int] = []
    result_bp: list[list[float]] = []
    result_bs: list[list[float]] = []
    result_ap: list[list[float]] = []
    result_as: list[list[float]] = []

    NAN = float("nan")
    N_LEVELS = 25

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, N_LEVELS), dtype=np.float64),
                np.empty((0, N_LEVELS), dtype=np.float64),
                np.empty((0, N_LEVELS), dtype=np.float64),
                np.empty((0, N_LEVELS), dtype=np.float64),
            )
        name = names[0]
        with zf.open(name, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = _loads(line)
                except Exception:
                    continue

                msg_type = rec.get("type", "")
                ts = int(rec.get("ts", 0))
                data = rec.get("data", {})
                bids = data.get("b", [])
                asks = data.get("a", [])

                if msg_type == "snapshot":
                    bid_book.clear()
                    ask_book.clear()
                    for entry in bids:
                        price = float(entry[0])
                        size = float(entry[1])
                        if size > 0.0:
                            bid_book[price] = size
                    for entry in asks:
                        price = float(entry[0])
                        size = float(entry[1])
                        if size > 0.0:
                            ask_book[price] = size
                    book_initialized = True
                elif msg_type == "delta" and book_initialized:
                    for entry in bids:
                        price = float(entry[0])
                        size = float(entry[1])
                        if size <= 0.0:
                            bid_book.pop(price, None)
                        else:
                            bid_book[price] = size
                    for entry in asks:
                        price = float(entry[0])
                        size = float(entry[1])
                        if size <= 0.0:
                            ask_book.pop(price, None)
                        else:
                            ask_book[price] = size
                else:
                    continue

                if not book_initialized:
                    continue
                if (ts - last_sample_ts) < ob_interval_ms:
                    continue

                sorted_bids = sorted(bid_book.items(), key=lambda x: -x[0])
                sorted_asks = sorted(ask_book.items(), key=lambda x: x[0])

                bp_row = [NAN] * N_LEVELS
                bs_row = [NAN] * N_LEVELS
                ap_row = [NAN] * N_LEVELS
                as_row = [NAN] * N_LEVELS

                for j in range(min(len(sorted_bids), N_LEVELS)):
                    bp_row[j] = sorted_bids[j][0]
                    bs_row[j] = sorted_bids[j][1]
                for j in range(min(len(sorted_asks), N_LEVELS)):
                    ap_row[j] = sorted_asks[j][0]
                    as_row[j] = sorted_asks[j][1]

                result_ts.append(ts)
                result_bp.append(bp_row)
                result_bs.append(bs_row)
                result_ap.append(ap_row)
                result_as.append(as_row)
                last_sample_ts = ts

    n = len(result_ts)
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, N_LEVELS), dtype=np.float64),
            np.empty((0, N_LEVELS), dtype=np.float64),
            np.empty((0, N_LEVELS), dtype=np.float64),
            np.empty((0, N_LEVELS), dtype=np.float64),
        )

    ts_ms = np.array(result_ts, dtype=np.int64)
    bid_prices = np.array(result_bp, dtype=np.float64)
    bid_sizes = np.array(result_bs, dtype=np.float64)
    ask_prices = np.array(result_ap, dtype=np.float64)
    ask_sizes = np.array(result_as, dtype=np.float64)

    try:
        np.savez(
            npz_path,
            ts_ms=ts_ms,
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
        )
    except Exception:
        pass

    return ts_ms, bid_prices, bid_sizes, ask_prices, ask_sizes


def load_ob_chunk(
    ob_zip_paths: list[str],
    dl_workers: int = 8,
    ob_interval_ms: int = 1_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = []

    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        futures = {
            pool.submit(_load_ob_day_arrays, path, ob_interval_ms): path
            for path in ob_zip_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                if len(result[0]) > 0:
                    arrays.append(result)
            except Exception as e:
                print(f"[warn] OB parse failed: {os.path.basename(path)}: {e}")

    if not arrays:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 25), dtype=np.float64),
            np.empty((0, 25), dtype=np.float64),
            np.empty((0, 25), dtype=np.float64),
            np.empty((0, 25), dtype=np.float64),
        )

    total = sum(len(a[0]) for a in arrays)
    ts_all = np.empty(total, dtype=np.int64)
    bp_all = np.empty((total, 25), dtype=np.float64)
    bs_all = np.empty((total, 25), dtype=np.float64)
    ap_all = np.empty((total, 25), dtype=np.float64)
    a_s_all = np.empty((total, 25), dtype=np.float64)

    offset = 0
    for ts, bp, bs, ap, a_s in arrays:
        n = len(ts)
        ts_all[offset:offset + n] = ts
        bp_all[offset:offset + n] = bp
        bs_all[offset:offset + n] = bs
        ap_all[offset:offset + n] = ap
        a_s_all[offset:offset + n] = a_s
        offset += n

    order = np.argsort(ts_all)
    ts_all = ts_all[order]
    bp_all = bp_all[order]
    bs_all = bs_all[order]
    ap_all = ap_all[order]
    a_s_all = a_s_all[order]
    ensure_monotonic(ts_all, "orderbook ts_ms")

    return ts_all, bp_all, bs_all, ap_all, a_s_all


def load_all_ob_as_dataframe(
    ob_zip_paths: list[str],
    dl_workers: int = 8,
    ob_interval_ms: int = 10_000,
) -> pd.DataFrame:
    arrays = []

    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        futures = {
            pool.submit(_load_ob_day_arrays, path, ob_interval_ms): path
            for path in ob_zip_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                if len(result[0]) > 0:
                    arrays.append(result)
            except Exception as e:
                print(f"[warn] OB parse failed: {os.path.basename(path)}: {e}")

    if not arrays:
        return pd.DataFrame()

    total = sum(len(a[0]) for a in arrays)
    ts_all = np.empty(total, dtype=np.int64)
    bp_all = np.empty((total, 25), dtype=np.float64)
    bs_all = np.empty((total, 25), dtype=np.float64)
    ap_all = np.empty((total, 25), dtype=np.float64)
    a_s_all = np.empty((total, 25), dtype=np.float64)

    offset = 0
    for ts, bp, bs, ap, a_s in arrays:
        n = len(ts)
        ts_all[offset:offset + n] = ts
        bp_all[offset:offset + n] = bp
        bs_all[offset:offset + n] = bs
        ap_all[offset:offset + n] = ap
        a_s_all[offset:offset + n] = a_s
        offset += n

    order = np.argsort(ts_all)
    ts_all = ts_all[order]
    bp_all = bp_all[order]
    bs_all = bs_all[order]
    ap_all = ap_all[order]
    a_s_all = a_s_all[order]
    ensure_monotonic(ts_all, "orderbook ts_ms")

    df = pd.DataFrame({"ts_ms": ts_all})
    for i in range(25):
        df[f"bid_price_{i}"] = bp_all[:, i]
        df[f"bid_size_{i}"] = bs_all[:, i]
        df[f"ask_price_{i}"] = ap_all[:, i]
        df[f"ask_size_{i}"] = a_s_all[:, i]
    return df


def _list_dates(start_date: _dt.date, end_date: _dt.date) -> list[_dt.date]:
    out: list[_dt.date] = []
    d = start_date
    while d < end_date:
        out.append(d)
        d += _dt.timedelta(days=1)
    return out


def _date_to_ms_utc(d: _dt.date) -> int:
    dt = _dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc)
    return int(dt.timestamp() * 1000)


def _load_trade_arrays_for_dates(
    trade_path_map: Dict[_dt.date, str],
    dates: list[_dt.date],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for d in dates:
        path = trade_path_map.get(d)
        if not path:
            continue
        try:
            arr = _read_bybit_trade_arrays(path)
            if len(arr[0]) > 0:
                chunks.append(arr)
        except Exception as e:
            print(f"[warn] trade parse failed ({d}): {e}")

    if not chunks:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )

    n = sum(len(x[0]) for x in chunks)
    ts_all = np.empty(n, dtype=np.int64)
    price_all = np.empty(n, dtype=np.float64)
    qty_all = np.empty(n, dtype=np.float64)
    ibm_all = np.empty(n, dtype=bool)

    off = 0
    for ts, price, qty, ibm in chunks:
        m = len(ts)
        ts_all[off:off + m] = ts
        price_all[off:off + m] = price
        qty_all[off:off + m] = qty
        ibm_all[off:off + m] = ibm
        off += m

    order = np.argsort(ts_all, kind="mergesort")
    ts_all = ts_all[order]
    price_all = price_all[order]
    qty_all = qty_all[order]
    ibm_all = ibm_all[order]
    ensure_monotonic(ts_all, "trade ts_ms")
    return ts_all, price_all, qty_all, ibm_all


def train_and_eval(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict:
    y_tr = df_train["y"].to_numpy(dtype=float)
    y_te = df_test["y"].to_numpy(dtype=float)

    mask_tr = np.isfinite(y_tr)
    mask_te = np.isfinite(y_te)

    X_train = df_train.loc[mask_tr, feature_cols].to_numpy(dtype=float)
    y_train = y_tr[mask_tr]

    X_test = df_test.loc[mask_te, feature_cols].to_numpy(dtype=float)
    y_test = y_te[mask_te]

    if len(y_train) < 10 or len(y_test) < 10:
        return {
            "spearman": float("nan"),
            "sign_acc": float("nan"),
            "mr_pnl": float("nan"),
            "mr_direction": float("nan"),
            "avg_hold_bars": float("nan"),
            "n_flip_short": 0,
            "n_flip_long": 0,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "feature_importance": [],
            "y_pred": None,
            "pred_index": None,
        }

    X_train, X_test, _ = _fill_all_nan_cols(X_train, X_test)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    use_es = (args.lgbm_early_stopping > 0) and (args.lgbm_boosting != "dart")
    n = len(y_train)
    if use_es:
        n_valid = max(int(n * float(args.lgbm_valid_frac)), 1)
        if n_valid < 10 or n - n_valid < 10:
            use_es = False
    if use_es:
        split = n - n_valid
        X_tr, y_tr2 = X_train_imp[:split], y_train[:split]
        X_va, y_va = X_train_imp[split:], y_train[split:]
    else:
        X_tr, y_tr2 = X_train_imp, y_train
        X_va = y_va = None

    model = lgb.LGBMRegressor(
        boosting_type=args.lgbm_boosting,
        objective="regression",
        learning_rate=args.lgbm_learning_rate,
        n_estimators=args.lgbm_n_estimators,
        num_leaves=args.lgbm_num_leaves,
        max_depth=args.lgbm_max_depth,
        min_child_samples=args.lgbm_min_data_in_leaf,
        colsample_bytree=args.lgbm_feature_fraction,
        subsample=args.lgbm_bagging_fraction,
        subsample_freq=args.lgbm_bagging_freq,
        reg_alpha=args.lgbm_lambda_l1,
        reg_lambda=args.lgbm_lambda_l2,
        min_split_gain=args.lgbm_min_gain_to_split,
        max_bin=args.lgbm_max_bin,
        random_state=args.seed,
        verbose=args.lgbm_verbose,
        drop_rate=args.lgbm_drop_rate,
        skip_drop=args.lgbm_skip_drop,
        max_drop=args.lgbm_max_drop,
        uniform_drop=bool(args.lgbm_uniform_drop),
    )

    if use_es:
        model.fit(
            X_tr,
            y_tr2,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(args.lgbm_early_stopping, verbose=False)],
        )
    else:
        model.fit(X_tr, y_tr2)

    y_pred = model.predict(X_test_imp)

    spearman = _spearman(y_pred, y_test)
    sign_acc = _sign_acc(y_pred, y_test)
    _test_index = df_test.index[mask_te]
    _y_pred_out = y_pred.copy()

    if "taker_imb" in df_test.columns:
        imb_test = df_test.loc[mask_te, "taker_imb"].to_numpy(dtype=float)
    else:
        imb_test = np.zeros_like(y_test)
    mr_leg = -np.sign(imb_test) * y_test
    mr_pnl = float(np.nanmean(mr_leg)) if len(mr_leg) > 0 else float("nan")
    mr_direction = float(np.nanmean(mr_leg > 0)) if len(mr_leg) > 0 else float("nan")

    if "hold_bars" in df_test.columns:
        hb = df_test.loc[mask_te, "hold_bars"].to_numpy(dtype=float)
        avg_hold = float(np.nanmean(hb)) if np.any(np.isfinite(hb)) else float("nan")
        n_flip_short = int(np.sum((hb >= 1) & (hb <= 3)))
        n_flip_long = int(np.sum(hb >= 4))
    else:
        avg_hold = float("nan")
        n_flip_short = 0
        n_flip_long = 0

    feat_imp = []
    try:
        booster = model.booster_
        if booster is not None:
            imp = booster.feature_importance(importance_type="gain")
            total = float(np.sum(imp))
            if total > 0:
                imp = imp / total
            order = np.argsort(imp)[::-1]
            for idx in order[:20]:
                feat_imp.append((feature_cols[idx], float(imp[idx])))
    except Exception:
        pass

    return {
        "spearman": float(spearman),
        "sign_acc": float(sign_acc),
        "mr_pnl": mr_pnl,
        "mr_direction": mr_direction,
        "avg_hold_bars": avg_hold,
        "n_flip_short": n_flip_short,
        "n_flip_long": n_flip_long,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_importance": feat_imp,
        "y_pred": _y_pred_out,
        "pred_index": _test_index,
    }


def compute_composite_event_scores(
    df_grid: pd.DataFrame,
    pctrank_window: int = 500,
    imb_K: int = 8,
) -> pd.DataFrame:
    """
    Compute 4-domain composite event score for each grid point.
    """
    minp = max(10, int(pctrank_window) // 10)
    imb_minp = max(2, int(imb_K) // 2)

    spread_col = "ob_spread_rel_close"
    if spread_col in df_grid.columns:
        spread = df_grid[spread_col].ffill().fillna(0.0)
    else:
        spread = pd.Series(0.0, index=df_grid.index)
        print("[WARN] ob_spread_rel_close not found; spread_score=0.5")
    spread_score = spread.rolling(pctrank_window, min_periods=minp).rank(pct=True)

    dt = df_grid["dt_sec"].clip(lower=0.0)
    dt_pctrank = dt.rolling(pctrank_window, min_periods=minp).rank(pct=True)
    activity_score = 1.0 - dt_pctrank

    imb_sign = np.sign(df_grid["taker_imb"])
    raw_consistency = (
        imb_sign
        .rolling(imb_K, min_periods=imb_minp)
        .sum()
        .abs()
        / float(imb_K)
    )
    imb_score = raw_consistency.rolling(pctrank_window, min_periods=minp).rank(pct=True)

    if "vpin_delta" in df_grid.columns:
        vpin_abs = df_grid["vpin_delta"].abs()
    else:
        vpin_abs = pd.Series(0.0, index=df_grid.index)
        print("[WARN] vpin_delta not found; vpin_score=0.5")
    vpin_score = vpin_abs.rolling(pctrank_window, min_periods=minp).rank(pct=True)

    composite = (spread_score + activity_score + imb_score + vpin_score) / 4.0

    df_grid["spread_score"] = spread_score.astype(np.float32)
    df_grid["activity_score"] = activity_score.astype(np.float32)
    df_grid["imb_score"] = imb_score.astype(np.float32)
    df_grid["vpin_score"] = vpin_score.astype(np.float32)
    df_grid["composite_score"] = composite.astype(np.float32)
    return df_grid


def calibrate_composite_threshold(
    composite: np.ndarray,
    target_rate: float = 0.07,
    search_min: float = 0.3,
    search_max: float = 0.9,
    n_steps: int = 60,
) -> float:
    """
    Find threshold that makes composite event rate closest to target.
    """
    valid = composite[np.isfinite(composite)]
    if len(valid) == 0:
        return 0.5

    best_thr = 0.5
    best_diff = float("inf")
    for thr in np.linspace(search_min, search_max, int(n_steps)):
        rate = float((valid >= thr).sum()) / float(len(valid))
        diff = abs(rate - float(target_rate))
        if diff < best_diff:
            best_diff = diff
            best_thr = float(thr)
    return best_thr


def run_walkforward_composite(
    df_feat: pd.DataFrame,
    trade_cols: list[str],
    ob_cols: list[str],
    args: argparse.Namespace,
) -> list[dict]:
    """Walk-forward A/B with composite score event filtering per fold."""
    all_cols = list(dict.fromkeys(trade_cols + ob_cols))

    folds = _build_wf_folds_bars(
        len(df_feat),
        args.wf_train_bars,
        args.wf_oos_bars,
        args.wf_step_bars,
        args.wf_gap_bars,
    )

    auto_purge = max(1, int(getattr(args, "max_hold_bars", 1)))
    if args.wf_purge_bars < 0:
        purge_bars = auto_purge
    else:
        purge_bars = max(int(args.wf_purge_bars), auto_purge)

    composite_all = df_feat["composite_score"].to_numpy(dtype=np.float64)

    results: list[dict] = []
    for fold_idx, (tr_start, tr_end, gap_end, os_end) in enumerate(folds):
        tr_end_purged = max(tr_start + 1, tr_end - purge_bars)

        train_comp = composite_all[tr_start:tr_end_purged]
        threshold = calibrate_composite_threshold(
            train_comp,
            target_rate=args.event_rate,
        )
        train_events = composite_all[tr_start:tr_end_purged] >= threshold
        oos_events = composite_all[gap_end:os_end] >= threshold

        df_train_full = df_feat.iloc[tr_start:tr_end_purged]
        df_test_full = df_feat.iloc[gap_end:os_end]
        df_train = df_train_full.iloc[train_events[:len(df_train_full)]]
        df_test = df_test_full.iloc[oos_events[:len(df_test_full)]]

        n_train_events = int(train_events.sum())
        n_oos_events = int(oos_events.sum())
        n_total_train = max(len(train_events), 1)
        n_total_oos = max(len(oos_events), 1)
        train_rate = n_train_events / n_total_train
        oos_rate = n_oos_events / n_total_oos

        oos_event_mask = oos_events[:len(df_test_full)]
        if oos_event_mask.sum() > 0:
            ev_rows = df_test_full.iloc[oos_event_mask]
            mean_spr = float(ev_rows["spread_score"].mean()) if "spread_score" in ev_rows else float("nan")
            mean_act = float(ev_rows["activity_score"].mean()) if "activity_score" in ev_rows else float("nan")
            mean_imb = float(ev_rows["imb_score"].mean()) if "imb_score" in ev_rows else float("nan")
            mean_vpn = float(ev_rows["vpin_score"].mean()) if "vpin_score" in ev_rows else float("nan")
        else:
            mean_spr = mean_act = mean_imb = mean_vpn = float("nan")

        res_trade = train_and_eval(df_train, df_test, trade_cols, args)
        res_full = train_and_eval(df_train, df_test, all_cols, args)
        delta = res_full["spearman"] - res_trade["spearman"]

        results.append(
            {
                "fold": fold_idx + 1,
                "n_train": res_trade["n_train"],
                "n_test": res_trade["n_test"],
                "n_train_events": n_train_events,
                "n_oos_events": n_oos_events,
                "train_event_rate": round(train_rate, 4),
                "oos_event_rate": round(oos_rate, 4),
                "threshold": round(threshold, 4),
                "mean_spread_score": round(mean_spr, 3),
                "mean_activity_score": round(mean_act, 3),
                "mean_imb_score": round(mean_imb, 3),
                "mean_vpin_score": round(mean_vpn, 3),
                "spearman_trade": res_trade["spearman"],
                "spearman_full": res_full["spearman"],
                "spearman_delta": delta,
                "sign_acc_trade": res_trade["sign_acc"],
                "sign_acc_full": res_full["sign_acc"],
                "feat_imp_full": res_full["feature_importance"],
            }
        )

        print(
            f"[WF] Fold {fold_idx+1:3d}/{len(folds)}: "
            f"thr={threshold:.2f} ev={oos_rate:.1%}({n_oos_events}) "
            f"[spr={mean_spr:.2f} act={mean_act:.2f} imb={mean_imb:.2f} vpn={mean_vpn:.2f}]  "
            f"Spear(T)={res_trade['spearman']:.4f}  "
            f"Spear(T+OB)={res_full['spearman']:.4f}  "
            f"Delta={delta:+.4f}"
        )

    return results


def compute_domain_score(df_grid: pd.DataFrame, pctrank_window: int = 2000) -> pd.DataFrame:
    """6-factor domain score used by v4 event filtering."""
    minp = max(50, int(pctrank_window) // 10)
    imb = df_grid["taker_imb"].to_numpy(dtype=np.float64)
    imb_sign = np.sign(imb)
    N = len(imb)

    streak = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        if imb_sign[i] == imb_sign[i - 1] and imb_sign[i] != 0:
            streak[i] = streak[i - 1] + 1
    s_streak = pd.Series(streak).rolling(pctrank_window, min_periods=minp).rank(pct=True).to_numpy()

    ob = df_grid["ob_imb_l1_close"].to_numpy(dtype=np.float64)
    ob_opp_raw = np.where(imb > 0, -ob, np.where(imb < 0, ob, 0.0))
    s_ob_opp = pd.Series(ob_opp_raw).rolling(pctrank_window, min_periods=minp).rank(pct=True).to_numpy()

    vol = df_grid["cur_vol_bp"].to_numpy(dtype=np.float64)
    s_low_vol = 1.0 - pd.Series(vol).rolling(pctrank_window, min_periods=minp).rank(pct=True).to_numpy()

    abs_imb_r = pd.Series(np.abs(imb)).rolling(pctrank_window, min_periods=minp).rank(pct=True).to_numpy()
    ret = df_grid["ret1_bp"].to_numpy(dtype=np.float64)
    abs_ret_r = pd.Series(np.abs(ret)).rolling(pctrank_window, min_periods=minp).rank(pct=True).to_numpy()
    s_absorption = np.clip(abs_imb_r - abs_ret_r, 0, 1)

    hours = pd.to_datetime(df_grid["ts_ms"], unit="ms").dt.hour.to_numpy()
    s_hours = np.isin(hours, [3, 4, 5, 9, 10, 11, 12, 21, 22, 23]).astype(np.float64)

    ret_prev1 = np.roll(ret, 1)
    ret_prev2 = np.roll(ret, 2)
    ret_prev1[0] = 0.0
    ret_prev2[:2] = 0.0
    s_3consec = (
        (np.abs(ret) > 3)
        & (np.abs(ret_prev1) > 3)
        & (np.abs(ret_prev2) > 3)
        & (np.sign(ret) == np.sign(ret_prev1))
        & (np.sign(ret) == np.sign(ret_prev2))
    ).astype(np.float64)

    total = np.zeros(N, dtype=np.float64)
    for s in [s_streak, s_ob_opp, s_low_vol, s_absorption, s_hours, s_3consec]:
        valid = np.isfinite(s)
        total[valid] += s[valid]

    df_grid["score_streak"] = s_streak.astype(np.float32)
    df_grid["score_ob_opp"] = s_ob_opp.astype(np.float32)
    df_grid["score_low_vol"] = s_low_vol.astype(np.float32)
    df_grid["score_absorption"] = s_absorption.astype(np.float32)
    df_grid["score_good_hours"] = s_hours.astype(np.float32)
    df_grid["score_3consec"] = s_3consec.astype(np.float32)
    df_grid["domain_score"] = total.astype(np.float32)
    return df_grid


def calibrate_domain_threshold(scores: np.ndarray, target_rate: float = 0.10) -> float:
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return 3.0
    return float(np.percentile(valid, (1.0 - target_rate) * 100.0))


if nn is not None:
    class OnlineMLP(nn.Module):
        """Lightweight MLP for prequential online learning."""
        def __init__(self, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)
else:
    class OnlineMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not installed. pip install torch")


def run_walkforward_v4(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], pd.Series]:
    """Single-model walk-forward with domain score filtering.

    Returns:
        results: Fold-level performance metrics.
        pred_series: Prediction series aligned to df_feat index.
    """
    folds = _build_wf_folds_bars(
        len(df_feat),
        args.wf_train_bars,
        args.wf_oos_bars,
        args.wf_step_bars,
        args.wf_gap_bars,
    )

    purge_bars = int(args.wf_purge_bars) if args.wf_purge_bars >= 0 else int(args.max_hold_bars)
    purge_bars = max(purge_bars, int(args.max_hold_bars))

    scores_all = df_feat["domain_score"].to_numpy(dtype=np.float64)

    results: list[dict] = []
    pred_dict: dict = {}
    for fold_idx, (tr_start, tr_end, gap_end, os_end) in enumerate(folds):
        tr_end_purged = max(tr_start + 1, tr_end - purge_bars)
        train_scores = scores_all[tr_start:tr_end_purged]
        threshold = calibrate_domain_threshold(train_scores, target_rate=args.event_rate)

        train_events = scores_all[tr_start:tr_end_purged] >= threshold
        oos_events = scores_all[gap_end:os_end] >= threshold

        df_train_full = df_feat.iloc[tr_start:tr_end_purged]
        df_test_full = df_feat.iloc[gap_end:os_end]
        df_train = df_train_full.iloc[train_events[:len(df_train_full)]]
        df_test = df_test_full.iloc[oos_events[:len(df_test_full)]]

        n_train_events = int(train_events.sum())
        n_oos_events = int(oos_events.sum())
        n_total_train = max(len(train_events), 1)
        n_total_oos = max(len(oos_events), 1)
        train_rate = n_train_events / n_total_train
        oos_rate = n_oos_events / n_total_oos

        res = train_and_eval(df_train, df_test, feature_cols, args)
        if res.get("pred_index") is not None and res.get("y_pred") is not None:
            for idx, val in zip(res["pred_index"], res["y_pred"]):
                pred_dict[idx] = float(val)

        results.append(
            {
                "fold": fold_idx + 1,
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "n_train_events": n_train_events,
                "n_oos_events": n_oos_events,
                "train_event_rate": round(train_rate, 4),
                "oos_event_rate": round(oos_rate, 4),
                "threshold": round(threshold, 4),
                "spearman": res["spearman"],
                "sign_acc": res["sign_acc"],
                "mr_pnl": res["mr_pnl"],
                "mr_direction": res["mr_direction"],
                "avg_hold_bars": res["avg_hold_bars"],
                "n_flip_short": res["n_flip_short"],
                "n_flip_long": res["n_flip_long"],
                "feat_imp": res["feature_importance"],
            }
        )

        print(
            f"[WF] Fold {fold_idx+1:3d}/{len(folds)}: "
            f"thr={threshold:.2f} ev={oos_rate:.1%}({n_oos_events}) "
            f"Spear={res['spearman']:.4f} MR_dir={res['mr_direction']:.3f} "
            f"MR_pnl={res['mr_pnl']:.3f} hold={res['avg_hold_bars']:.2f}"
        )
    pred_series = pd.Series(pred_dict, dtype=float, name="y_pred")
    pred_series = pred_series.reindex(df_feat.index)
    return results, pred_series


def run_walkforward_online_mlp(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], pd.Series]:
    """
    Prequential online MLP walk-forward.

    구조:
    1. 초기 train 윈도우(wf_train_bars)로 MLP 초기 학습.
       - 정규화 통계(mean/std)를 float64로 초기 train 구간에서 계산 후 고정.
       - X_all 전체를 사전 정규화하여 float32로 저장 (메모리 절약).
       - domain_score threshold는 루프 전에 rolling quantile로 전 구간 사전계산
         (pctrank_window 기준, 매 봉 즉시 반영).
    2. wf_train_bars 시점 이후 각 봉 t마다:
       a. threshold_arr[t] 조회 (사전계산값).
       b. domain_score >= threshold이면 예측 수행 (prequential: 먼저 예측).
       c. t - max_hold_bars 시점의 샘플 y가 확정됐고 domain event이면
          replay buffer에서 mini-batch 샘플링 후 새 샘플 추가, gradient step.
    3. fold 경계는 평가 집계용으로만 사용. fold 경계에서 모델 초기화 없음.
    4. fold별 threshold는 OOS 예측 시점 threshold들의 평균으로 기록.
    """
    if torch is None:
        raise RuntimeError("torch not installed. pip install torch")

    import collections

    scores_all = df_feat["domain_score"].to_numpy(dtype=np.float64)
    y_all = df_feat["y"].to_numpy(dtype=np.float64)
    X_all = df_feat[feature_cols].to_numpy(dtype=np.float64)

    n_total = len(df_feat)
    max_hold = int(args.max_hold_bars)
    train_end = int(args.wf_train_bars)
    pctrank_w = int(args.pctrank_window)

    scores_series = pd.Series(scores_all)
    threshold_arr = (
        scores_series
        .rolling(pctrank_w, min_periods=1)
        .quantile(1.0 - args.event_rate)
        .to_numpy()
    )

    X_init = X_all[:train_end]
    feat_mean = np.nanmean(X_init, axis=0)
    feat_std = np.nanstd(X_init, axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    def normalize(X: np.ndarray) -> np.ndarray:
        return (X - feat_mean) / feat_std

    def to_tensor(X: np.ndarray) -> "torch.Tensor":
        X = np.where(np.isfinite(X), X, 0.0)
        return torch.tensor(X, dtype=torch.float32)

    purge_bars = max(
        int(args.max_hold_bars),
        int(args.wf_purge_bars) if args.wf_purge_bars >= 0 else int(args.max_hold_bars),
    )
    tr_end_purged = max(1, train_end - purge_bars)
    threshold = calibrate_domain_threshold(
        scores_all[:tr_end_purged], target_rate=args.event_rate
    )

    input_dim = len(feature_cols)
    model = OnlineMLP(input_dim=input_dim, hidden_dim=args.mlp_hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)

    init_mask = (
        np.isfinite(y_all[:tr_end_purged]) &
        (scores_all[:tr_end_purged] >= threshold)
    )
    X_init_ev = normalize(X_all[:tr_end_purged][init_mask])
    y_init_ev = y_all[:tr_end_purged][init_mask]

    if len(y_init_ev) >= 10:
        model.train()
        ds = torch.utils.data.TensorDataset(
            to_tensor(X_init_ev),
            torch.tensor(y_init_ev, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        for _ in range(5):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    print(
        f"[OnlineMLP] Init train: {int(init_mask.sum())} events, "
        f"init_threshold={threshold:.3f} (rolling pctrank_window={pctrank_w})"
    )

    replay_buffer: collections.deque = collections.deque(maxlen=args.mlp_replay_size)

    folds = _build_wf_folds_bars(
        n_total,
        args.wf_train_bars,
        args.wf_oos_bars,
        args.wf_step_bars,
        args.wf_gap_bars,
    )
    fold_os_ranges = [(gap_end, os_end) for _, _, gap_end, os_end in folds]

    fold_preds: list[list] = [[] for _ in folds]
    fold_trues: list[list] = [[] for _ in folds]
    fold_imbs: list[list] = [[] for _ in folds]
    fold_holds: list[list] = [[] for _ in folds]
    fold_thresholds: list[list] = [[] for _ in folds]

    pred_dict: dict = {}

    taker_imb_all = (
        df_feat["taker_imb"].to_numpy(dtype=np.float64)
        if "taker_imb" in df_feat.columns
        else np.zeros(n_total)
    )
    hold_bars_all = (
        df_feat["hold_bars"].to_numpy(dtype=np.float64)
        if "hold_bars" in df_feat.columns
        else np.full(n_total, np.nan)
    )

    model.eval()

    for t in range(train_end, n_total):
        threshold = threshold_arr[t]

        if scores_all[t] >= threshold and np.isfinite(y_all[t]):
            x_t = normalize(X_all[t:t + 1])
            with torch.no_grad():
                y_hat = float(model(to_tensor(x_t)).item())
            pred_dict[df_feat.index[t]] = y_hat

            for fi, (gs, ge) in enumerate(fold_os_ranges):
                if gs <= t < ge:
                    fold_preds[fi].append(y_hat)
                    fold_trues[fi].append(y_all[t])
                    fold_imbs[fi].append(taker_imb_all[t])
                    fold_holds[fi].append(hold_bars_all[t])
                    fold_thresholds[fi].append(threshold)
                    break

        conf_t = t - max_hold
        if conf_t >= train_end and conf_t >= 0:
            if (
                scores_all[conf_t] >= threshold
                and np.isfinite(y_all[conf_t])
                and np.all(np.isfinite(X_all[conf_t]))
            ):
                x_conf = normalize(X_all[conf_t])
                y_conf = float(y_all[conf_t])
                batch_xs = [x_conf]
                batch_ys = [y_conf]
                if len(replay_buffer) >= 2:
                    buf_list = list(replay_buffer)
                    n_replay = min(args.mlp_batch_size - 1, len(buf_list))
                    idxs = np.random.choice(len(buf_list), n_replay, replace=False)
                    for i in idxs:
                        rx, ry = buf_list[i]
                        batch_xs.append(rx)
                        batch_ys.append(ry)

                replay_buffer.append((x_conf, y_conf))

                X_b = to_tensor(np.stack(batch_xs))
                y_b = torch.tensor(batch_ys, dtype=torch.float32)

                model.train()
                optimizer.zero_grad()
                loss = F.mse_loss(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.eval()

    results: list[dict] = []
    for fi, (gs, ge) in enumerate(fold_os_ranges):
        yp = np.array(fold_preds[fi])
        yt = np.array(fold_trues[fi])
        imb = np.array(fold_imbs[fi])
        hb = np.array(fold_holds[fi])

        if len(yp) < 10:
            sp = float("nan")
            sign_acc = float("nan")
            mr_pnl = float("nan")
            mr_dir = float("nan")
        else:
            sp = float(_spearman(yp, yt))
            sign_acc = float(_sign_acc(yp, yt))
            mr_leg = -np.sign(imb) * yt
            mr_pnl = float(np.nanmean(mr_leg))
            mr_dir = float(np.nanmean(mr_leg > 0))

        avg_hold = float(np.nanmean(hb)) if np.any(np.isfinite(hb)) else float("nan")
        avg_thr = float(np.mean(fold_thresholds[fi])) if fold_thresholds[fi] else float("nan")
        oos_n = int(len(yp))
        oos_rate = oos_n / max(ge - gs, 1)

        results.append({
            "fold": fi + 1,
            "n_train": int(init_mask.sum()),
            "n_test": oos_n,
            "n_train_events": int(init_mask.sum()),
            "n_oos_events": oos_n,
            "train_event_rate": round(float(init_mask.mean()), 4),
            "oos_event_rate": round(oos_rate, 4),
            "threshold": round(avg_thr, 4),
            "spearman": sp,
            "sign_acc": sign_acc,
            "mr_pnl": mr_pnl,
            "mr_direction": mr_dir,
            "avg_hold_bars": avg_hold,
            "n_flip_short": 0,
            "n_flip_long": 0,
            "feat_imp": [],
        })

        print(
            f"[OnlineMLP] Fold {fi+1:3d}/{len(folds)}: "
            f"ev={oos_n} "
            f"Spear={sp:.4f} MR_dir={mr_dir:.3f} "
            f"MR_pnl={mr_pnl:.3f} hold={avg_hold:.2f}"
        )

    pred_series = pd.Series(pred_dict, dtype=float, name="y_pred")
    pred_series = pred_series.reindex(df_feat.index)
    return results, pred_series


def _process_single_grid(
    T_ms: int,
    v_target: float,
    n_lags: int,
    ts: np.ndarray,
    price: np.ndarray,
    qty: np.ndarray,
    ibm: np.ndarray,
    cumsum: np.ndarray,
    ob_ts: np.ndarray,
    raw_feats: dict,
    mid_prices: np.ndarray,
) -> dict | None:
    """Compute one grid-point feature row. Thread-safe (numba releases GIL)."""
    df_bars = build_bars_at_grid(
        ts,
        price,
        qty,
        ibm,
        int(T_ms),
        float(v_target),
        int(n_lags) + 1,
        qty_cumsum=cumsum,
    )
    if df_bars is None:
        return None

    feat = compute_grid_base_features(df_bars, int(n_lags))
    if len(ob_ts) > 0:
        ob_feat = attach_ob_features(
            df_bars,
            ob_ts,
            raw_feats,
            mid_prices,
        )
        feat.update(format_ob_features(ob_feat, int(n_lags)))

    feat["ts_ms"] = int(T_ms)
    feat["y"] = np.nan
    feat["hold_bars"] = np.nan
    return feat


def print_report(results: list[dict], args: argparse.Namespace) -> None:
    print("=" * 80)
    print("  ob_poc_v4: Walk-Forward Results (Variable Horizon + Domain Score)")
    print(f"  Symbol: {args.symbol} (Bybit) | Period: {args.start} to {args.end}")
    print(
        f"  grid={args.grid_interval_sec}s | max_hold_bars={args.max_hold_bars} | "
        f"v_target={args.v_target} | n_lags={args.n_lags}"
    )
    if getattr(args, "model_type", "lgbm") == "online_mlp":
        print(
            f"  model=online_mlp | hidden={args.mlp_hidden_dim} | lr={args.mlp_lr} | "
            f"replay={args.mlp_replay_size} | batch={args.mlp_batch_size} | "
            f"train={args.wf_train_bars} oos={args.wf_oos_bars}"
        )
    else:
        print(
            f"  feat_scale={args.feat_scale} | {args.lgbm_boosting}/{args.lgbm_n_estimators} | "
            f"train={args.wf_train_bars} oos={args.wf_oos_bars}"
        )
    print(f"  domain score: event_rate={args.event_rate} | pctrank_window={args.pctrank_window}")
    print("=" * 80)
    print("Fold | Ev_train | Ev_test |  thr  | Spear   | SignAcc | MR_dir | MR_pnl | Hold")

    for r in results:
        print(
            f"{r['fold']:4d} | {r.get('n_train_events', '?'):8} | {r.get('n_oos_events', '?'):7} | "
            f"{r.get('threshold', 0):5.3f} | {r['spearman']:7.4f} | {r['sign_acc']:7.4f} | "
            f"{r['mr_direction']:6.3f} | {r['mr_pnl']:6.3f} | {r['avg_hold_bars']:4.2f}"
        )

    spears = [r["spearman"] for r in results]
    signs = [r["sign_acc"] for r in results]
    mr_dirs = [r["mr_direction"] for r in results]
    mr_pnls = [r["mr_pnl"] for r in results]
    holds = [r["avg_hold_bars"] for r in results]
    ev_rates = [r.get("oos_event_rate", float("nan")) for r in results]
    thresholds = [r.get("threshold", float("nan")) for r in results]

    print("-" * 80)
    print(f"Mean Spearman:    {np.nanmean(spears):.4f}")
    print(f"Median Spearman:  {np.nanmedian(spears):.4f}")
    print(f"Std Spearman:     {np.nanstd(spears):.4f}")
    print(f"Mean SignAcc:     {np.nanmean(signs):.4f}")
    print(f"Mean MR_dir:      {np.nanmean(mr_dirs):.3f}")
    print(f"Mean MR_pnl:      {np.nanmean(mr_pnls):.3f}")
    print(f"Mean Hold Bars:   {np.nanmean(holds):.2f}")
    print(f"Mean event rate (OOS): {np.nanmean(ev_rates):.1%}")
    print(f"Mean threshold: {np.nanmean(thresholds):.3f}")

    if results:
        print("=== Feature Importance (last fold) ===")
        last = results[-1].get("feat_imp", [])
        ob_set = set(build_ob_feature_cols(args.n_lags))
        print("Rank | Feature                     | Importance")
        for i, (feat, imp) in enumerate(last, 1):
            tag = " [OB]" if feat in ob_set else ""
            print(f"{i:4d} | {feat:27s} | {imp:10.3f}{tag}")


def save_json_report(results: list[dict], args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    spears = [r.get("spearman", np.nan) for r in results]
    signs = [r.get("sign_acc", np.nan) for r in results]
    mr_dirs = [r.get("mr_direction", np.nan) for r in results]
    mr_pnls = [r.get("mr_pnl", np.nan) for r in results]
    holds = [r.get("avg_hold_bars", np.nan) for r in results]

    summary = {
        "n_folds": len(results),
        "mean_spearman": float(np.nanmean(spears)) if results else float("nan"),
        "median_spearman": float(np.nanmedian(spears)) if results else float("nan"),
        "std_spearman": float(np.nanstd(spears)) if results else float("nan"),
        "mean_sign_acc": float(np.nanmean(signs)) if results else float("nan"),
        "mean_mr_direction": float(np.nanmean(mr_dirs)) if results else float("nan"),
        "mean_mr_pnl": float(np.nanmean(mr_pnls)) if results else float("nan"),
        "mean_hold_bars": float(np.nanmean(holds)) if results else float("nan"),
        "mean_oos_event_rate": float(np.nanmean([r.get("oos_event_rate", np.nan) for r in results])) if results else float("nan"),
        "mean_threshold": float(np.nanmean([r.get("threshold", np.nan) for r in results])) if results else float("nan"),
        "total_folds": len(results),
    }

    out = {
        "config": {
            "model_type": getattr(args, "model_type", "lgbm"),
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "grid_interval_sec": args.grid_interval_sec,
            "max_hold_bars": args.max_hold_bars,
            "v_target": args.v_target,
            "n_lags": args.n_lags,
            "feat_scale": args.feat_scale,
            "pctrank_window": args.pctrank_window,
            "ob_interval_ms": args.ob_interval_ms,
            "event_rate": args.event_rate,
            "wf_train_bars": args.wf_train_bars,
            "wf_oos_bars": args.wf_oos_bars,
            "wf_step_bars": args.wf_step_bars,
            "wf_gap_bars": args.wf_gap_bars,
            "wf_purge_bars": args.wf_purge_bars,
            "lgbm_boosting": args.lgbm_boosting,
            "lgbm_learning_rate": args.lgbm_learning_rate,
            "lgbm_n_estimators": args.lgbm_n_estimators,
            "lgbm_num_leaves": args.lgbm_num_leaves,
            "lgbm_max_depth": args.lgbm_max_depth,
            "lgbm_min_data_in_leaf": args.lgbm_min_data_in_leaf,
            "lgbm_feature_fraction": args.lgbm_feature_fraction,
            "lgbm_bagging_fraction": args.lgbm_bagging_fraction,
            "lgbm_bagging_freq": args.lgbm_bagging_freq,
            "lgbm_lambda_l2": args.lgbm_lambda_l2,
            **(
                {
                    "mlp_hidden_dim": args.mlp_hidden_dim,
                    "mlp_lr": args.mlp_lr,
                    "mlp_replay_size": args.mlp_replay_size,
                    "mlp_batch_size": args.mlp_batch_size,
                }
                if getattr(args, "model_type", "lgbm") == "online_mlp"
                else {}
            ),
        },
        "summary": summary,
        "folds": [
            {
                "fold": r["fold"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "n_train_events": r.get("n_train_events"),
                "n_oos_events": r.get("n_oos_events"),
                "train_event_rate": r.get("train_event_rate"),
                "oos_event_rate": r.get("oos_event_rate"),
                "threshold": r.get("threshold"),
                "spearman": r.get("spearman"),
                "sign_acc": r.get("sign_acc"),
                "mr_pnl": r.get("mr_pnl"),
                "mr_direction": r.get("mr_direction"),
                "avg_hold_bars": r.get("avg_hold_bars"),
                "n_flip_short": r.get("n_flip_short"),
                "n_flip_long": r.get("n_flip_long"),
            }
            for r in results
        ],
    }

    out_path = os.path.join(args.outdir, "ob_poc_v4_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ob_poc_v4: Time-grid domain-score system")

    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD exclusive")
    p.add_argument("--cache_dir", default="./.cache_ob_v4")
    p.add_argument("--outdir", default="./out_ob_v4")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--dl_workers", type=int, default=4)

    p.add_argument("--grid_interval_sec", type=int, default=60, help="Grid interval in seconds")
    p.add_argument("--v_target", type=float, default=250.0, help="Volume bar size by qty")
    p.add_argument("--n_lags", type=int, default=2, help="Uses n_lags+1 bars at each grid")
    p.add_argument("--max_hold_bars", type=int, default=15, help="Maximum variable-target holding bars")

    p.add_argument("--feat_scale", type=int, default=18, help="Rolling feature window in grid units")
    p.add_argument("--pctrank_window", type=int, default=2000, help="Percent-rank window in grid units")
    p.add_argument("--ob_interval_ms", type=int, default=1000)

    p.add_argument("--wf_train_bars", type=int, default=100000, help="Train grid count")
    p.add_argument("--wf_oos_bars", type=int, default=25000, help="OOS grid count")
    p.add_argument("--wf_step_bars", type=int, default=25000)
    p.add_argument("--wf_gap_bars", type=int, default=0)
    p.add_argument("--wf_purge_bars", type=int, default=-1, help="-1 = auto (=max_hold_bars)")
    p.add_argument(
        "--grid_workers",
        type=int,
        default=4,
        help="Number of parallel workers for grid computation (0=sequential)",
    )

    p.add_argument(
        "--event_rate",
        type=float,
        default=0.10,
        help="Target event rate for domain-score threshold calibration (0.10=10%%)",
    )

    p.add_argument("--lgbm_boosting", default="dart", choices=["dart", "gbdt"])
    p.add_argument("--lgbm_learning_rate", type=float, default=0.02)
    p.add_argument("--lgbm_n_estimators", type=int, default=1000)
    p.add_argument("--lgbm_num_leaves", type=int, default=31)
    p.add_argument("--lgbm_max_depth", type=int, default=8)
    p.add_argument("--lgbm_min_data_in_leaf", type=int, default=250)
    p.add_argument("--lgbm_feature_fraction", type=float, default=0.5)
    p.add_argument("--lgbm_bagging_fraction", type=float, default=0.8)
    p.add_argument("--lgbm_bagging_freq", type=int, default=1)
    p.add_argument("--lgbm_lambda_l1", type=float, default=0.0)
    p.add_argument("--lgbm_lambda_l2", type=float, default=10.0)
    p.add_argument("--lgbm_min_gain_to_split", type=float, default=0.0)
    p.add_argument("--lgbm_drop_rate", type=float, default=0.1)
    p.add_argument("--lgbm_skip_drop", type=float, default=0.5)
    p.add_argument("--lgbm_max_drop", type=int, default=50)
    p.add_argument("--lgbm_max_bin", type=int, default=255)
    p.add_argument("--lgbm_early_stopping", type=int, default=0)
    p.add_argument("--lgbm_valid_frac", type=float, default=0.15)
    p.add_argument("--lgbm_uniform_drop", type=int, default=0)
    p.add_argument("--lgbm_verbose", type=int, default=-1)

    p.add_argument(
        "--model_type",
        default="lgbm",
        choices=["lgbm", "online_mlp"],
        help="Model type: lgbm (default) or online_mlp (prequential online learning)",
    )
    p.add_argument(
        "--mlp_hidden_dim", type=int, default=128,
        help="Hidden layer size for online MLP",
    )
    p.add_argument(
        "--mlp_lr", type=float, default=3e-4,
        help="Adam learning rate for online MLP",
    )
    p.add_argument(
        "--mlp_replay_size", type=int, default=5000,
        help="Experience replay buffer size",
    )
    p.add_argument(
        "--mlp_batch_size", type=int, default=32,
        help="Mini-batch size for online MLP update (new sample + replay)",
    )
    p.add_argument(
        "--mlp_threshold_recalib_interval", type=int, default=1000,
        help="How often (in bars) to recalibrate domain_score threshold for online MLP",
    )

    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "model_type", "lgbm") == "online_mlp" and torch is None:
        raise RuntimeError("torch not installed. pip install torch")
    if getattr(args, "model_type", "lgbm") != "online_mlp" and lgb is None:
        raise RuntimeError("lightgbm not installed; pip install lightgbm")
    if nb is None:
        raise RuntimeError("numba not installed; pip install numba")
    if args.v_target <= 0:
        raise ValueError("--v_target must be > 0")
    if args.grid_interval_sec <= 0:
        raise ValueError("--grid_interval_sec must be > 0")
    if args.max_hold_bars <= 0:
        raise ValueError("--max_hold_bars must be > 0")
    if args.n_lags < 0:
        raise ValueError("--n_lags must be >= 0")
    if args.feat_scale < 5:
        raise ValueError("--feat_scale must be >= 5")
    if args.pctrank_window < 20:
        raise ValueError("--pctrank_window must be >= 20")
    if not (1 <= args.dl_workers <= 32):
        raise ValueError("--dl_workers must be 1-32")
    if not (0.0 < args.lgbm_feature_fraction <= 1.0):
        raise ValueError("--lgbm_feature_fraction must be in (0, 1]")
    if not (0.0 < args.lgbm_bagging_fraction <= 1.0):
        raise ValueError("--lgbm_bagging_fraction must be in (0, 1]")
    if args.lgbm_num_leaves < 2:
        raise ValueError("--lgbm_num_leaves must be >= 2")
    if args.lgbm_min_data_in_leaf < 1:
        raise ValueError("--lgbm_min_data_in_leaf must be >= 1")
    if not (0.01 <= args.event_rate <= 0.50):
        raise ValueError("--event_rate must be between 0.01 and 0.50")
    if args.grid_workers < 0:
        raise ValueError("--grid_workers must be >= 0")

    start_date = _dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = _dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    if end_date <= start_date:
        raise ValueError("--end must be after --start")


def main() -> None:
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)

    start_date = _dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = _dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    trade_dates = _list_dates(start_date, end_date)

    print(f"[Data] Downloading trades: {args.start} to {args.end}")
    trade_paths: list[tuple[_dt.date, str]] = []
    with ThreadPoolExecutor(max_workers=args.dl_workers) as pool:
        futures = {
            pool.submit(
                download_bybit_trades,
                args.symbol,
                d,
                args.cache_dir,
                args.timeout,
                args.max_retries,
            ): d
            for d in trade_dates
        }
        for future in as_completed(futures):
            d = futures[future]
            try:
                trade_paths.append((d, future.result()))
            except Exception as e:
                print(f"[warn] {d}: trade download failed: {e}")

    trade_paths.sort(key=lambda x: x[0])
    trade_path_map: Dict[_dt.date, str] = {d: p for d, p in trade_paths}
    if not trade_path_map:
        print("[ERROR] No trade files available.")
        sys.exit(1)
    print(f"[Data] Trades: {len(trade_path_map)} days downloaded")

    print(f"[Data] Downloading orderbook: {args.start} to {args.end}")
    ob_paths = download_bybit_ob(
        args.symbol,
        args.start,
        args.end,
        args.cache_dir,
        args.timeout,
        args.max_retries,
        dl_workers=args.dl_workers,
    )

    ob_path_map: Dict[_dt.date, str] = {}
    for pth in ob_paths:
        b = os.path.basename(pth)
        if len(b) >= 10:
            try:
                d = _dt.datetime.strptime(b[:10], "%Y-%m-%d").date()
                ob_path_map[d] = pth
            except Exception:
                pass

    avail_dates = sorted(trade_path_map.keys())
    first_ts = None
    for d in avail_dates:
        ts, _, _, _ = _read_bybit_trade_arrays(trade_path_map[d])
        if len(ts) > 0:
            first_ts = int(ts[0])
            break
    last_ts = None
    for d in reversed(avail_dates):
        ts, _, _, _ = _read_bybit_trade_arrays(trade_path_map[d])
        if len(ts) > 0:
            last_ts = int(ts[-1])
            break

    if first_ts is None or last_ts is None or last_ts <= first_ts:
        print("[ERROR] Could not determine valid trade timestamp range.")
        sys.exit(1)

    grid_interval_ms = int(args.grid_interval_sec * 1000)
    warmup_ms = 12 * 3600 * 1000
    grid_start_raw = (first_ts // grid_interval_ms) * grid_interval_ms
    grid_start = grid_start_raw + warmup_ms
    grid_start = ((grid_start + grid_interval_ms - 1) // grid_interval_ms) * grid_interval_ms
    grid_end = ((last_ts // grid_interval_ms) + 1) * grid_interval_ms
    grid_times = np.arange(grid_start, grid_end, grid_interval_ms, dtype=np.int64)
    print(
        f"[Grid] interval={args.grid_interval_sec}s, "
        f"warmup={warmup_ms // 1000}s, points={len(grid_times):,}"
    )

    BATCH_DAYS = 15
    OVERLAP_BACK_DAYS = 2
    OVERLAP_FWD_DAYS = 0
    all_grid_rows: list[Dict[str, float]] = []

    for batch_idx in range(0, len(trade_dates), BATCH_DAYS):
        batch_dates = trade_dates[batch_idx:batch_idx + BATCH_DAYS]
        if not batch_dates:
            continue

        overlap_back = trade_dates[max(0, batch_idx - OVERLAP_BACK_DAYS):batch_idx]
        fwd_end = min(len(trade_dates), batch_idx + BATCH_DAYS + OVERLAP_FWD_DAYS)
        overlap_fwd = trade_dates[batch_idx + BATCH_DAYS:fwd_end]
        load_dates = overlap_back + batch_dates + overlap_fwd

        ts_batch, price_batch, qty_batch, ibm_batch = _load_trade_arrays_for_dates(trade_path_map, load_dates)
        if len(ts_batch) == 0:
            print(f"[Grid] Batch {batch_idx // BATCH_DAYS + 1}: no trades")
            continue
        qty_cumsum = np.cumsum(qty_batch, dtype=np.float64)

        ob_load_paths = [ob_path_map[d] for d in load_dates if d in ob_path_map]
        if ob_load_paths:
            ob_ts, bp, bs, ap, a_s = load_ob_chunk(
                ob_load_paths,
                dl_workers=args.dl_workers,
                ob_interval_ms=args.ob_interval_ms,
            )
            if len(ob_ts) > 0:
                raw_feats, mid_prices = compute_ob_raw_features_vectorized(bp, bs, ap, a_s)
            else:
                raw_feats = {k: np.empty(0, dtype=float) for k in OB_RAW_FEATURES}
                mid_prices = np.empty(0, dtype=float)
            del bp, bs, ap, a_s
            gc.collect()
        else:
            ob_ts = np.empty(0, dtype=np.int64)
            raw_feats = {k: np.empty(0, dtype=float) for k in OB_RAW_FEATURES}
            mid_prices = np.empty(0, dtype=float)

        batch_start_ts = _date_to_ms_utc(batch_dates[0])
        batch_end_ts = _date_to_ms_utc(batch_dates[-1] + _dt.timedelta(days=1))
        grid_mask = (grid_times >= batch_start_ts) & (grid_times < batch_end_ts)
        grid_in_batch = grid_times[grid_mask]

        if len(grid_in_batch) == 0:
            del ts_batch, price_batch, qty_batch, ibm_batch, qty_cumsum
            del ob_ts, raw_feats, mid_prices
            gc.collect()
            continue

        built = 0
        if args.grid_workers <= 0:
            for T in grid_in_batch:
                df_bars = build_bars_at_grid(
                    ts_batch,
                    price_batch,
                    qty_batch,
                    ibm_batch,
                    int(T),
                    args.v_target,
                    args.n_lags + 1,
                    qty_cumsum=qty_cumsum,
                )
                if df_bars is None:
                    continue

                feat = compute_grid_base_features(df_bars, args.n_lags)
                if len(ob_ts) > 0:
                    ob_feat = attach_ob_features(df_bars, ob_ts, raw_feats, mid_prices)
                    feat.update(format_ob_features(ob_feat, args.n_lags))

                feat["ts_ms"] = int(T)
                feat["y"] = np.nan
                feat["hold_bars"] = np.nan
                all_grid_rows.append(feat)
                built += 1
        else:
            from functools import partial

            worker_fn = partial(
                _process_single_grid,
                v_target=args.v_target,
                n_lags=args.n_lags,
                ts=ts_batch,
                price=price_batch,
                qty=qty_batch,
                ibm=ibm_batch,
                cumsum=qty_cumsum,
                ob_ts=ob_ts,
                raw_feats=raw_feats,
                mid_prices=mid_prices,
            )
            with ThreadPoolExecutor(max_workers=args.grid_workers) as pool:
                futures = [pool.submit(worker_fn, int(T)) for T in grid_in_batch]
                for fut in futures:
                    feat = fut.result()
                    if feat is not None:
                        all_grid_rows.append(feat)
                        built += 1

        print(
            f"[Grid] Batch {batch_idx // BATCH_DAYS + 1}: "
            f"{len(grid_in_batch):,} grids, {built:,} rows"
        )

        del ts_batch, price_batch, qty_batch, ibm_batch, qty_cumsum
        del ob_ts, raw_feats, mid_prices
        gc.collect()

    if not all_grid_rows:
        print("[ERROR] No grid feature rows were built.")
        sys.exit(1)

    df_grid = pd.DataFrame(all_grid_rows)
    df_grid = df_grid.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"], keep="last").reset_index(drop=True)

    grid_close = df_grid["close"].to_numpy(dtype=np.float64)
    grid_imb = df_grid["taker_imb"].to_numpy(dtype=np.float64)
    y_var, hold_bars_arr = compute_variable_target_vec(grid_close, grid_imb, args.max_hold_bars)
    df_grid["y"] = y_var.astype(np.float32)
    df_grid["hold_bars"] = hold_bars_arr.astype(np.float32)

    df_grid = compute_rolling_features(df_grid, args.feat_scale, args.pctrank_window)
    ob_pctrank_base = [
        "ob_spread_close",
        "ob_spread_rel_close",
        "ob_total_depth_5_close",
        "ob_microprice_adj_close",
    ]
    _pctrank_minp = max(10, int(args.pctrank_window) // 10)
    for col in ob_pctrank_base:
        if col in df_grid.columns:
            df_grid[f"{col}_pctrank"] = (
                df_grid[col]
                .rolling(int(args.pctrank_window), min_periods=_pctrank_minp)
                .rank(pct=True)
            )

    # Ensure lag columns used by v4 feature list are populated on the grid time series.
    lag_fill_cols = [
        "taker_imb", "pde", "kyle_lambda", "roll_spread",
        "ret1_bp", "range_bp", "close_pos", "close_vs_vwap",
        "large_imb_share", "entropy_norm", "burst_cv",
        "trade_count", "mid_size_share", "rev_count_ratio",
        "par", "twi", "vpin_delta", "tca",
        "ob_imb_l1_close", "ob_imb_l5_close", "ob_imb_l10_close",
        "ob_depth_ratio_5_close", "ob_depth_ratio_10_close",
        "ob_spread_close", "ob_spread_rel_close",
        "ob_microprice_adj_close", "ob_total_depth_5_close",
    ]
    for base in lag_fill_cols:
        if base not in df_grid.columns:
            continue
        for lag in range(1, int(args.n_lags) + 1):
            lag_col = f"{base}_lag{lag}"
            shifted = df_grid[base].shift(lag)
            if lag_col in df_grid.columns:
                df_grid[lag_col] = df_grid[lag_col].where(df_grid[lag_col].notna(), shifted)
            else:
                df_grid[lag_col] = shifted

    df_grid = compute_domain_score(df_grid, args.pctrank_window)

    feature_cols = build_v4_feature_cols(args.n_lags)
    for col in feature_cols + ["ts_ms", "y", "hold_bars", "domain_score"]:
        if col not in df_grid.columns:
            df_grid[col] = np.nan

    n_score_valid = int(np.isfinite(df_grid["domain_score"].to_numpy()).sum())
    n_y_valid = int(np.isfinite(df_grid["y"].to_numpy()).sum())
    print(f"[Domain] valid scores: {n_score_valid:,} / {len(df_grid):,}")
    print(f"[Target] y valid: {n_y_valid:,} / {len(df_grid):,}")
    print(f"[Target] y describe: {df_grid['y'].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).to_dict()}")

    ob_match = float(np.mean(np.isfinite(df_grid.get("ob_imb_l1_close", pd.Series(np.nan, index=df_grid.index)).to_numpy())))
    print(f"[Diag] OB match rate: {ob_match*100:.2f}%")

    nan_ratio = df_grid[feature_cols + ["y", "hold_bars"]].isna().mean().sort_values(ascending=False)
    print(f"[Diag] Top NaN ratio cols: {nan_ratio.head(10).to_dict()}")

    os.makedirs(args.outdir, exist_ok=True)
    feat_path = os.path.join(args.outdir, "df_grid_v4.parquet")
    df_grid.to_parquet(feat_path)
    print(f"[Diag] Saved grid features: {feat_path} ({len(df_grid):,} rows)")

    print(
        f"[WF] Starting walk-forward "
        f"(train={args.wf_train_bars}, oos={args.wf_oos_bars}, step={args.wf_step_bars})"
    )
    if args.model_type == "online_mlp":
        results, pred_series = run_walkforward_online_mlp(df_grid, feature_cols, args)
    else:
        results, pred_series = run_walkforward_v4(df_grid, feature_cols, args)

    if not results:
        print("[ERROR] No folds. Need more data or smaller wf_train_bars.")
        sys.exit(1)

    df_pred = df_grid[["ts_ms", "y", "hold_bars", "taker_imb", "domain_score"]].copy()
    df_pred["y_pred"] = pred_series

    folds_info = _build_wf_folds_bars(
        len(df_grid),
        args.wf_train_bars,
        args.wf_oos_bars,
        args.wf_step_bars,
        args.wf_gap_bars,
    )
    df_pred["fold"] = -1
    fold_col = df_pred.columns.get_loc("fold")
    for fold_idx, (_, _, gap_end, os_end) in enumerate(folds_info):
        df_pred.iloc[gap_end:os_end, fold_col] = fold_idx + 1

    pred_path = os.path.join(args.outdir, "df_pred_v4.parquet")
    df_pred.to_parquet(pred_path)
    print(f"[Diag] Saved predictions: {pred_path} ({df_pred['y_pred'].notna().sum():,} predicted rows)")

    print_report(results, args)
    save_json_report(results, args)


if __name__ == "__main__":
    main()
