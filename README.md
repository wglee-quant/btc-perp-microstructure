# BTC Perpetual Futures Microstructure Analysis

Research on short-term price prediction in Bitcoin perpetual futures markets using order flow microstructure data.

## Overview

This project analyzes BTC perpetual futures market microstructure to build a short-term directional prediction system. The core approach combines information asymmetry measurement with order flow dynamics.

## Key Components

### Domain Scoring System
- Built on the **Glosten-Milgrom model** for measuring information asymmetry between informed and uninformed traders
- Integrates order flow imbalance, trade arrival patterns, and spread dynamics into a unified domain score

### Signal Generation (RT3 Trigger)
- Custom trigger mechanism that generates trading signals based on domain score thresholds
- Designed to capture regime shifts in market microstructure

### Validation
- **Walk-Forward Validation** to prevent look-ahead bias
- **Direction Accuracy: 59.7%**
- Statistically significant **Spearman Correlation**

## Key Finding

Pure microstructure-based signals alone struggle to overcome **taker fees (7.5–11bp)**. This limitation motivated the next phase of research: integrating macro indicators and sentiment data through an AI-powered pipeline.

## Tech Stack
- Python (data collection & analysis)
- Binance API (market data)
- LightGBM (ML modeling)
- Walk-Forward cross-validation framework

## Next Steps
→ See [Macro AI Trading Pipeline] (upcoming) — combining macro indicators, market sentiment, and microstructure data with local LLMs for cost-efficient analysis.

## Author
**Woonggyu Lee** — High school student & AI/Quantitative Finance Researcher
- 📧 leewoonggyu@outlook.kr
- 💼 [LinkedIn](https://www.linkedin.com/in/woonggyu-lee/)

---
*This research was conducted independently. Feedback from industry professionals including derivatives traders and quant engineers helped shape the research direction.*
