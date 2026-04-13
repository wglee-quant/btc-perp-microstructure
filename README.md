# BTC Perpetual Futures Microstructure Research

**Selective trading through market microstructure signals on BTC-USDT perpetual futures.**

> Core finding: Price *direction* is a martingale (AC1 ≈ 0.003), but *volatility* is predictable. Instead of forecasting direction, we identify high-opportunity microstructure regimes and trade only when conditions align.

---

## Motivation

Most crypto trading strategies attempt to predict price direction — a near-random process on short horizons. This research takes a different approach: rather than asking "where will price go?", we ask **"when does the market reveal exploitable structure?"**

Using order book microstructure data from Binance & Bybit BTC-USDT perpetual futures (5.4B ticks, 2025 full year), we develop a selective-entry system that:

- Detects **adverse selection regimes** via spread dynamics (RT3 Trigger)
- Assesses **multi-dimensional market state** through a composite score (Domain Score)
- Validates through rigorous **walk-forward cross-validation** (up to 42-fold)

## Methodology

### 1. Volume Bars & Feature Engineering

We replace time-based bars with **volume bars** (constant-volume sampling), normalizing market activity. From each bar we extract **87 features** (22 base + 65 lag across 5 categories), all normalized via rolling percentile rank (window=2000).

### 2. RT3 Trigger — Adverse Selection Event Detection

Inspired by **Glosten-Milgrom (1985)**, the RT3 trigger identifies moments when market makers widen spreads due to informed order flow.

| Condition | Threshold | Interpretation |
|-----------|-----------|----------------|
| `spread_pct` | ≥ 80th percentile (rolling) | Elevated spread = informed trading |
| `imb_consistency` | ≥ 0.75 (K=8 bars) | Sustained directional order flow |

**Key discovery**: The trigger captures event *completion*, not initiation — the dominant signal is **mean reversion**, not momentum. K=4→K=8 transition improved Spearman from 0.103 to 0.133 (+143% PnL).

**Walk-Forward 8-fold**: All folds positive (p < 0.0001), SP range 0.071–0.178, mean 0.133.

### 3. Domain Score — Multi-Factor Market State

Built on top of RT3, the Domain Score converts discrete on/off into a continuous 0–6 scale using 6 microstructure elements:

| Element | Captures |
|---------|----------|
| Spread expansion | Adverse selection cost |
| Fast trading | Information flow intensity |
| Order flow persistence | Directional pressure |
| Flow shock | Sudden information asymmetry |
| Book-trade divergence | Passive vs. aggressive order mismatch |
| Low volatility | Mean reversion strength |

**Monotonic relationship**: Score 0 (SP=-0.006, random) → Score 5 (SP=-0.257, strong MR). At score ≥ 4: z=-10.9, p=6.5×10⁻²⁸.

**Walk-Forward 16-fold**: SP=0.130, 16/16 positive folds, directional accuracy 59.7%.

### 4. Rolling Spearman Post-Filter

Real-time quality filter using trailing 200-event Spearman. SP ≥ 0.10 threshold: **SP 0.130 → 0.212**, trading 61% of events.

### 5. Hold Period Structure

| Duration | Spearman | Signal |
|----------|----------|--------|
| 1–3 bars | -0.52 | Strong mean reversion |
| 25–61 bars | +0.32 | Trend following (sign flip) |

Short-term: market-maker liquidity dominates. Long-term: accumulated information drives trends.

## Key Results

| Configuration | Sharpe (WF) | Dir. Accuracy | Significance |
|--------------|-------------|---------------|-------------|
| RT3 (8-fold) | **0.133** | 54.2% | p < 0.0001 |
| Domain Score (16-fold) | **0.130** | **59.7%** | z=-10.9 |
| + Rolling SP filter | **0.212** | — | 61% coverage |

## Repository Contents

| File | Description |
|------|-------------|
| `vortexbar_lab.py` | RT3 trigger logic, volume bar feature engineering (4,781 lines) |
| `ob_poc_v4.py` | Domain score computation, walk-forward framework (3,190 lines) |
| `BTC_Perp_Research_Brief_v2.pdf` | Detailed research report (Korean, 5 pages) |

> **Note**: Model weights, PnL data, and live-trading infrastructure are not included.

## Technical Stack

- **Data**: Binance & Bybit BTCUSDT perp (aggTrades + 100ms L2 orderbook snapshots)
- **ML**: LightGBM DART
- **Features**: 87 microstructure features, rolling percentile normalization
- **Validation**: Expanding-window walk-forward (8/16/42-fold)

## References

- Glosten, L. R., & Milgrom, P. R. (1985). *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders.* JFE, 14(1), 71-100.
- Engle, R. F., & Russell, J. R. (1998). *Autoregressive Conditional Duration.* Econometrica, 66(5), 1127-1162.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

## Contact

**Woonggyu Lee** — Independent quantitative researcher, 17 y/o  
Email: leewoonggyu@outlook.kr  
GitHub: [@wglee-quant](https://github.com/wglee-quant)
