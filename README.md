# Pattern Dataset Generator (H&S / DT/DB / TT/TB)

This project generates labeled datasets of technical patterns using a deterministic pipeline:
- Download OHLCV via yfinance
- Compute ZigZag pivots
- Validate and score patterns using rule-based checks with indicators (RSI, Stochastic, MACD, OBV, ATR, Volume)
- Export a CSV for labeling and modeling

## Supported patterns
- H&S (OCO) and Inverse H&S (OCOI)
- Double Top (DT) and Double Bottom (DB)
- Triple Top (TT) and Triple Bottom (TB)

## Quick start
```
pip install -r requirements.txt
python src/patterns/OCOs/necklineconfirmada.py --patterns ALL
```

Filter by tickers, strategies, intervals, and period:
```
python src/patterns/OCOs/necklineconfirmada.py \
  --tickers BTC-USD,ETH-USD \
  --strategies swing_short,intraday_momentum \
  --intervals 15m,1h \
  --period 2y \
  --patterns HNS,DTB,TTB
```

Output: `data/datasets/patterns_by_strategy/dataset_patterns_final.csv`
- Includes columns: `ticker`,`timeframe`,`strategy`,`padrao_tipo`,`score_total`, plus `valid_*` flags, pivot fields (`*_idx`,`*_preco`), and convenience fields: `tipo`,`score`,`pivos` (JSON list of pivots).

## Configuration (Config)
All thresholds and weights are centralized in `src/patterns/OCOs/necklineconfirmada.py` under `class Config`.
- RSI: `RSI_LENGTH`, `RSI_OVERBOUGHT`, `RSI_OVERSOLD`, `RSI_STRONG_OVERBOUGHT`, `RSI_STRONG_OVERSOLD`, `RSI_DIVERGENCE_MIN_DELTA`
- Stochastic: `STOCH_K`, `STOCH_D`, `STOCH_SMOOTH_K`, `STOCH_OVERBOUGHT`, `STOCH_OVERSOLD`, `STOCH_CROSS_LOOKBACK_BARS`, `STOCH_DIVERGENCE_MIN_DELTA`
- MACD: `MACD_FAST`, `MACD_SLOW`, `MACD_SIGNAL`, `MACD_SIGNAL_CROSS_LOOKBACK_BARS`
- Volume breakout: `VOLUME_BREAKOUT_LOOKBACK_BARS`, `VOLUME_BREAKOUT_MULTIPLIER`, `BREAKOUT_SEARCH_MAX_BARS`
- Retest ATR: `NECKLINE_RETEST_ATR_MULTIPLIER`
- ZigZag behavior: `ZIGZAG_EXTEND_TO_LAST_BAR`
- Scoring weights: `SCORE_WEIGHTS_HNS`, `SCORE_WEIGHTS_DTB`, `SCORE_WEIGHTS_TTB`, and respective `MINIMUM_SCORE_*`
- Debug: `DTB_DEBUG`, `HNS_DEBUG`, `TTB_DEBUG`

## Debugging
Enable verbose debug to see rule-by-rule decisions:
```
# In Config
DTB_DEBUG = True
TTB_DEBUG = True
```
This will print reasons for acceptance/rejection and final scores for DT/DB and TT/TB. HNS uses concise logging.

## Examples
- HNS only: `python src/patterns/OCOs/necklineconfirmada.py --patterns HNS`
- DT/DB only: `python src/patterns/OCOs/necklineconfirmada.py --patterns DTB`
- TT/TB only: `python src/patterns/OCOs/necklineconfirmada.py --patterns TTB`
- All patterns: `python src/patterns/OCOs/necklineconfirmada.py --patterns ALL`

## Notebook
See `notebooks/demo_patterns.ipynb` for a minimal demo detecting HNS, DTB, and TTB on small samples. 