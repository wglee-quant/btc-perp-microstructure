# BTC Perpetual Futures Microstructure Research

> **Selective trading through market microstructure signals on BTC-USDT perpetual futures.**  
> Walk-forward validated | 5.4B ticks | Glosten-Milgrom inspired

---

## TL;DR

Price direction in crypto is near-random (AC1 ≈ 0.003). Rather than forecasting direction, this research asks: **when does the market reveal exploitable structure?**

Using 5.4 billion aggTrade ticks from Binance & Bybit BTC-USDT perp (full year 2025), I built a selective-entry system that detects adverse-selection regimes and trades only when microstructure conditions align. The core pipeline — RT3 trigger → Domain Score → Walk-Forward CV → Rolling Spearman filter — achieves a walk-forward Spearman of **0.212** at 61% event coverage with all folds positive (p < 0.0001).

---

## The Problem

Most crypto strategies try to predict *where* price goes. On short horizons, this is a losing game — returns are near-i.i.d. and transaction costs eat everything.

The alternative: predict *when* the market is in a state where a specific edge exists. This shifts the problem from direction forecasting to **regime detection**, which is much more tractable given microstructure observables.

---

## Methodology

### 1. Volume Bars & Feature Engineering

Time-based bars conflate high-activity and low-activity periods. **Volume bars** (constant-volume sampling) normalize market activity, making features stationary across regimes.

From each bar: **87 features** — 22 base + 65 lags across 5 categories — all normalized via rolling percentile rank (window = 2,000 bars).

---

### 2. RT3 Trigger — Adverse Selection Event Detection

Inspired by **Glosten-Milgrom (1985)**: market makers widen spreads when informed order flow arrives. RT3 detects these moments.

| Condition | Threshold | Interpretation |
|-----------|-----------|----------------|
| `spread_pct` | ≥ 80th pct (rolling) | Elevated spread → informed trading |
| `imb_consistency` | ≥ 0.75 over K=8 bars | Sustained directional order flow |

**Key discovery**: The trigger fires at event *completion*, not initiation. The dominant signal is **mean reversion**, not momentum. Extending K from 4 → 8 improved Spearman ρ from 0.103 to 0.133 (+143% PnL).

**Walk-forward validation (8-fold, expanding window):** All 8 folds positive, ρ range 0.071–0.178, mean 0.133 (p < 0.0001).

---

### 3. Domain Score — Multi-Factor Market State

Converts RT3's binary on/off signal into a continuous **0–6 score** using six microstructure elements:

| Element | Captures |
|---------|----------|
| Spread expansion | Adverse selection cost |
| Fast trading | Information flow intensity |
| Order flow persistence | Directional pressure |
| Flow shock | Sudden information asymmetry |
| Book-trade divergence | Passive vs. aggressive order mismatch |
| Low volatility | Mean reversion strength |

**Monotonic relationship:** Score 0 → ρ = -0.006 (random) through Score 5 → ρ = -0.257 (strong MR).  
At score ≥ 4: z = -10.9, p = 6.5 × 10⁻²⁸.

**Walk-forward validation (16-fold):** ρ = 0.130, 16/16 folds positive, directional accuracy 59.7%.

---

### 4. Walk-Forward Cross-Validation

All results use **expanding-window walk-forward CV** (8 / 16 / 42-fold configurations), ensuring no future data leaks into training. The strictest configuration runs 42 folds — each fold trains on all prior data and tests on the next unseen window.

This methodology directly addresses the survival-bias and overfitting issues common in crypto backtests.

---

### 5. Rolling Spearman Post-Filter

A real-time quality gate: compute trailing 200-event Spearman ρ. Trade only when ρ ≥ 0.10.

| | Before filter | After filter |
|--|--------------|-------------|
| Spearman ρ | 0.130 | **0.212** |
| Event coverage | 100% | 61% |

The filter adapts dynamically — if the edge degrades in a given market regime, it steps aside automatically.

---

## Key Results

| Configuration | Spearman ρ (WF) | Dir. Accuracy | Significance |
|--------------|----------------|---------------|-------------|
| RT3 (8-fold WF) | 0.133 | 54.2% | p < 0.0001 |
| Domain Score (16-fold WF) | 0.130 | 59.7% | z = -10.9 |
| + Rolling SP filter | **0.212** | — | 61% coverage |

**Hold period structure:**

| Duration | Spearman | Interpretation |
|----------|----------|---------------|
| 1–3 bars | -0.52 | Strong mean reversion (market-maker liquidity dominates) |
| 25–61 bars | +0.32 | Trend following (accumulated information drives price) |

---

## Research Evolution

This repository documents **one thread** of a broader research program. In parallel, I have been exploring structural inefficiencies in altcoin/meme-coin perpetual futures — a different market microstructure with different dynamics.

**Meme-Coin Futures (61 coins × 30 days, Binance aggTrades):**  
The key discovery: volume-accompanied spikes exhibit structural mean reversion (+56.8 bps gross, ~60% win rate at vol_ratio ≥ 3). Direction asymmetry is strong — short after spike works, long after dip does not. A 5-axis discriminant analysis (order-flow shift, post-event price action, concurrent events, session, z-score) produced a best composite filter of **net +309 bps/trade, 93% win rate** (N=27 trades/month), with full front/back half cross-validation passing.

The broader pipeline under development:

```
Macro Signal Layer   →   Microstructure Regime Detection   →   Execution
(funding / basis /       (RT3 + Domain Score + Rolling SP)     (LightGBM DART
 altcoin correlation)                                           + WF validation)
```

---

## Repository Contents

| File | Description |
|------|-------------|
| `vortexbar_lab.py` | RT3 trigger, volume bar feature engineering (4,781 lines) |
| `ob_poc_v4.py` | Domain Score computation, walk-forward framework (3,190 lines) |
| `BTC_Perp_Research_Brief_v2.pdf` | Detailed research report (5 pages) |

> **Note**: Model weights, PnL curves, and live-trading infrastructure are not included.

---

## Technical Stack

- **Data**: Binance & Bybit BTCUSDT perp — aggTrades + 100ms L2 orderbook snapshots (5.4B ticks, 2025)
- **ML**: LightGBM DART
- **Features**: 87 microstructure features, rolling percentile normalization
- **Validation**: Expanding-window walk-forward CV (8 / 16 / 42-fold)
- **Stack**: Python — numpy, pandas, scikit-learn, lightgbm, numba, pyarrow

---

## About

**Woonggyu Lee** — Independent quantitative researcher, 17 y/o, South Korea.

I research market microstructure, ML-driven regime detection, and structural inefficiencies in crypto perpetual futures. Current focus: building a macro-to-micro AI trading pipeline that integrates funding rate signals, cross-asset basis, and altcoin correlation into a unified selective-entry framework.

- GitHub: [@wglee-quant](https://github.com/wglee-quant)  
- LinkedIn: *(coming soon)*  
- Email: leewoonggyu@outlook.kr

---

## Full Brief

A detailed writeup (methodology, feature definitions, validation protocol, and preliminary PnL analysis) is available as [`BTC_Perp_Research_Brief_v2.pdf`](./BTC_Perp_Research_Brief_v2.pdf) in this repository.

---

## References

- Glosten, L. R., & Milgrom, P. R. (1985). *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders.* JFE, 14(1), 71–100.
- Engle, R. F., & Russell, J. R. (1998). *Autoregressive Conditional Duration.* Econometrica, 66(5), 1127–1162.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
