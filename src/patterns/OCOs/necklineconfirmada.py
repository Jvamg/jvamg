"""Technical pattern dataset generator (H&S/Inverse H&S and DT/DB).

Pipeline: download OHLCV (yfinance) → compute ZigZag pivots → validate/score
patterns with deterministic rules → save final CSV for labeling/modeling.

Run as a script to filter by tickers/strategies/intervals.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from typing import List, Dict, Any, Optional, Tuple
import os
import time
import re
from colorama import Fore, Style, init
import argparse
import json

# Initialize Colorama
init(autoreset=True)


class Config:
    """Global configuration for universe, strategies, rules, and output."""
    TICKERS = [
        'AAVE-USD',  # Aave
        'ADA-USD',   # Cardano
        'ALGO-USD',  # Algorand
        'AVAX-USD',  # Avalanche
        'BCH-USD',   # Bitcoin Cash
        'BNB-USD',   # BNB
        'BTC-USD',   # Bitcoin
        'BSV-USD',   # Bitcoin SV
        'CHZ-USD',   # Chiliz
        'CRO-USD',   # Cronos
        'DOGE-USD',  # Dogecoin
        'DOT-USD',   # Polkadot
        'EGLD-USD',  # MultiversX
        'EOS-USD',   # EOS
        'ETC-USD',   # Ethereum Classic
        'ETH-USD',   # Ethereum
        'FIL-USD',   # Filecoin
        'FLOW-USD',  # Flow
        'HBAR-USD',  # Hedera
        'ICP-USD',   # Internet Computer
        'LDO-USD',   # Lido DAO
        'LINK-USD',  # Chainlink
        'LTC-USD',   # Litecoin
        'MANA-USD',  # Decentraland
        'MKR-USD',   # Maker
        'NEAR-USD',  # NEAR Protocol
        'NEO-USD',   # Neo
        'OP-USD',    # Optimism
        'QNT-USD',   # Quant
        'SHIB-USD',  # Shiba Inu
        'SNX-USD',   # Synthetix
        'SOL-USD',   # Solana
        'THETA-USD',  # Theta Network
        'TRX-USD',   # TRON
        'VET-USD',   # VeChain
        'XLM-USD',   # Stellar
        'XMR-USD',   # Monero
        'XRP-USD',   # XRP
        'XTZ-USD',   # Tezos
        'ZIL-USD',   # Zilliqa
    ]
    DATA_PERIOD = '5y'

    ZIGZAG_STRATEGIES = {
        # ------- SCALPING (micro-structures) ----------
        'scalping_aggressive': {
            '5m':  {'depth': 3, 'deviation': 0.25}
        },
        'scalping_moderate': {
            '5m':  {'depth': 4, 'deviation': 0.40},
            '15m': {'depth': 5, 'deviation': 0.60}
        },
        'scalping_conservative': {
            '5m':  {'depth': 5, 'deviation': 0.55},
            '15m': {'depth': 6, 'deviation': 0.75}
        },

        # ------- INTRADAY (15m-1h) ----------
        'intraday_momentum': {
            '5m':  {'depth': 6, 'deviation': 0.80},
            '15m': {'depth': 7, 'deviation': 1.10},
            '1h':  {'depth': 8, 'deviation': 1.60}
        },
        'intraday_range': {
            '5m':  {'depth': 7, 'deviation': 1.00},
            '15m': {'depth': 8, 'deviation': 1.30},
            '1h':  {'depth': 9, 'deviation': 1.90}
        },

        # ------- SWING (hours to days) ----------
        'swing_short': {
            '15m': {'depth': 8,  'deviation': 2.0},
            '1h':  {'depth': 10, 'deviation': 2.8},
            '4h':  {'depth': 12, 'deviation': 4.0}
        },
        'swing_medium': {
            '1h':  {'depth': 10, 'deviation': 3.2},
            '4h':  {'depth': 12, 'deviation': 4.8},
            '1d':  {'depth': 10, 'deviation': 6.0}
        },
        'swing_long': {
            '4h':  {'depth': 13, 'deviation': 5.0},
            '1d':  {'depth': 12, 'deviation': 7.0},
            '1wk': {'depth': 10, 'deviation': 8.5}
        },

        # ------- POSITION / MACRO ----------
        'position_trend': {
            '1d':  {'depth': 15, 'deviation': 9.0},
            '1wk': {'depth': 12, 'deviation': 12.0},
            '1mo': {'depth': 8,  'deviation': 15.0}
        },
        'macro_trend_primary': {
            '1wk': {'depth': 16, 'deviation': 13.0},
            '1mo': {'depth': 10, 'deviation': 18.0}
        }
    }

    # --- H&S SCORING/RULES ---
    SCORE_WEIGHTS_HNS = {
        # Mandatory rules
        'valid_extremo_cabeca': 20, 'valid_contexto_cabeca': 15,
        'valid_simetria_ombros': 10, 'valid_neckline_plana': 5,
        'valid_base_tendencia': 5,
        'valid_neckline_retest_p6': 15,
        # Optional confirmations (updated)
        'valid_divergencia_rsi': 10,
        'valid_divergencia_rsi_strong': 20,
        'valid_divergencia_macd': 10,
        'valid_macd_signal_cross': 8,
        'valid_estocastico_divergencia': 8,
        'valid_estocastico_cross': 5,
        'valid_ombro_direito_fraco': 5,
        'valid_perfil_volume': 5,
        'valid_volume_breakout_neckline': 10,
        'valid_proeminencia_cabeca': 10,
    }
    MINIMUM_SCORE_HNS = 70

    # --- FUTURE: DOUBLE TOP/BOTTOM (DT/DB) ---
    SCORE_WEIGHTS_DTB = {
        # Mandatory rules
        'valid_estrutura_picos_vales': 20,
        'valid_simetria_extremos': 10,
        'valid_profundidade_vale_pico': 15,
        'valid_contexto_extremos': 5,
        'valid_contexto_tendencia': 5,
        'valid_neckline_retest_p4': 15,
        # Optional confirmations (updated)
        'valid_perfil_volume_decrescente': 10,
        'valid_divergencia_obv': 15,
        'valid_divergencia_rsi': 8,
        'valid_divergencia_rsi_strong': 16,
        'valid_divergencia_macd': 5,
        'valid_macd_signal_cross': 5,
        'valid_estocastico_divergencia': 6,
        'valid_estocastico_cross': 5,
        'valid_segundo_topo_menor': 5,
        'valid_volume_breakout_neckline': 10,
    }
    MINIMUM_SCORE_DTB = 70
    DTB_SYMMETRY_TOLERANCE_FACTOR = 0.1
    DTB_VALLEY_PEAK_DEPTH_RATIO = 0.15
    DTB_TREND_MIN_DIFF_FACTOR = 0.05
    DTB_DEBUG = True
    DEBUG_DIR = 'logs'
    DTB_DEBUG_FILE = os.path.join(DEBUG_DIR, 'dtb_debug.log')

    # --- Triple Top/Bottom (TT/TB) scoring ---
    SCORE_WEIGHTS_TTB = {
        # Mandatory rules
        'valid_estrutura_picos_vales': 20,
        'valid_simetria_extremos': 10,
        'valid_profundidade_vale_pico': 15,
        'valid_contexto_extremos': 5,
        'valid_contexto_tendencia': 5,
        'valid_neckline_retest_p6': 15,
        # Optional confirmations (reuse DTB optional set)
        'valid_perfil_volume_decrescente': 10,
        'valid_divergencia_obv': 15,
        'valid_divergencia_rsi': 8,
        'valid_divergencia_rsi_strong': 16,
        'valid_divergencia_macd': 5,
        'valid_macd_signal_cross': 5,
        'valid_estocastico_divergencia': 6,
        'valid_estocastico_cross': 5,
        'valid_volume_breakout_neckline': 10,
    }
    MINIMUM_SCORE_TTB = 70

    # Indicator thresholds/config (centralized)
    RSI_LENGTH = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    RSI_STRONG_OVERBOUGHT = 80
    RSI_STRONG_OVERSOLD = 20
    RSI_DIVERGENCE_MIN_DELTA = 5.0

    STOCH_K = 14
    STOCH_D = 3
    STOCH_SMOOTH_K = 3
    STOCH_OVERBOUGHT = 80
    STOCH_OVERSOLD = 20
    STOCH_CROSS_LOOKBACK_BARS = 7
    STOCH_DIVERGENCE_MIN_DELTA = 5.0

    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MACD_SIGNAL_CROSS_LOOKBACK_BARS = 7

    VOLUME_BREAKOUT_LOOKBACK_BARS = 20
    VOLUME_BREAKOUT_MULTIPLIER = 1.8
    BREAKOUT_SEARCH_MAX_BARS = 60

    # Validation parameters
    HEAD_SIGNIFICANCE_RATIO = 1.1
    SHOULDER_SYMMETRY_TOLERANCE = 0.30
    NECKLINE_FLATNESS_TOLERANCE = 0.25
    HEAD_EXTREME_LOOKBACK_FACTOR = 2
    HEAD_EXTREME_LOOKBACK_MIN_BARS = 20

    RECENT_PATTERNS_LOOKBACK_COUNT = 1
    NECKLINE_RETEST_ATR_MULTIPLIER = 0.75

    ZIGZAG_EXTEND_TO_LAST_BAR = True
    # Debug toggles
    HNS_DEBUG = False
    TTB_DEBUG = True

    MAX_DOWNLOAD_TENTATIVAS, RETRY_DELAY_SEGUNDOS = 3, 5
    OUTPUT_DIR = 'data/datasets/patterns_by_strategy'
    FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, 'dataset_patterns_final.csv')

# --- Helper functions ---


def _dtb_debug(msg: str) -> None:
    """Prints DT/DB debug message and appends a sanitized copy to a single log file.

    - Respects Config.DTB_DEBUG (no-op if False)
    - Writes to Config.DTB_DEBUG_FILE (creates directory if needed)
    - Strips ANSI color codes for file output
    """
    if not getattr(Config, 'DTB_DEBUG', False):
        return
    # Print to console (keep colors)
    print(msg)
    try:
        os.makedirs(getattr(Config, 'DEBUG_DIR', 'logs'), exist_ok=True)
        sanitized = re.sub(r'\x1b\[[0-9;]*m', '', msg)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(getattr(Config, 'DTB_DEBUG_FILE', os.path.join('logs', 'dtb_debug.log')), 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {sanitized}\n")
    except Exception:
        # Silent fail for logging to avoid breaking pipeline
        pass


def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance respecting interval limits.

    Automatically adjusts ``period`` for intraday intervals and normalizes
    columns to lowercase.
    """
    original_period = period
    if 'mo' in interval:
        period = 'max'
    elif 'm' in interval:
        period = '7d'
    elif 'h' in interval:
        period = '2y'

    if period != original_period:
        print(f"{Fore.YELLOW}Notice: period '{original_period}' adjusted to '{period}' for interval '{interval}' to respect API limits.{Style.RESET_ALL}")

    for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
        try:
            df = yf.download(tickers=ticker, period=period,
                             interval=interval, auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [col.lower() for col in df.columns]
                # Ensure timezone-naive index to avoid mixing tz-aware/naive later
                try:
                    df.index = df.index.tz_localize(None)
                except Exception:
                    pass
                return df
            else:
                raise ValueError("Download returned an empty DataFrame.")
        except Exception as e:
            if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1:
                print(
                    f"{Fore.YELLOW}Attempt {tentativa + 1} failed. Retrying in {Config.RETRY_DELAY_SEGUNDOS}s...{Style.RESET_ALL}")
                time.sleep(Config.RETRY_DELAY_SEGUNDOS)
            else:
                raise ConnectionError(
                    f"Download failed for {ticker}/{interval} after {Config.MAX_DOWNLOAD_TENTATIVAS} attempts. Error: {e}")

    raise ConnectionError(
        f"Unexpected download failure for {ticker}/{interval}.")


def calcular_zigzag_oficial(df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
    """Calcula pivôs ZigZag com alternância e desvio percentual mínimo."""
    peak_series, valley_series = df['high'], df['low']
    window_size = 2 * depth + 1
    rolling_max, rolling_min = peak_series.rolling(window=window_size, center=True, min_periods=1).max(
    ), valley_series.rolling(window=window_size, center=True, min_periods=1).min()
    candidate_peaks_df, candidate_valleys_df = df[peak_series ==
                                                  rolling_max], df[valley_series == rolling_min]
    candidates = []
    for idx, row in candidate_peaks_df.iterrows():
        candidates.append(
            {'idx': idx, 'preco': row[peak_series.name], 'tipo': 'PICO'})
    for idx, row in candidate_valleys_df.iterrows():
        candidates.append(
            {'idx': idx, 'preco': row[valley_series.name], 'tipo': 'VALE'})
    candidates = sorted(
        list({p['idx']: p for p in candidates}.values()), key=lambda x: x['idx'])
    if len(candidates) < 2:
        return []
    confirmed_pivots = [candidates[0]]
    last_pivot = candidates[0]
    for i in range(1, len(candidates)):
        candidate = candidates[i]
        if candidate['tipo'] == last_pivot['tipo']:
            if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                confirmed_pivots[-1], last_pivot = candidate, candidate
            continue
        if last_pivot['preco'] == 0:
            continue
        price_dev = abs(
            candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
        if price_dev >= deviation_percent:
            confirmed_pivots.append(candidate)
            last_pivot = candidate
    if Config.ZIGZAG_EXTEND_TO_LAST_BAR and confirmed_pivots:
        last_confirmed_pivot = confirmed_pivots[-1]
        last_bar = df.iloc[-1]

        # Se o último candle prolongar o movimento na MESMA direção,
        # apenas atualizamos o pivô existente. Caso contrário, criamos um novo pivô.
        if last_confirmed_pivot['tipo'] == 'PICO':
            # Movimento ainda de alta: atualiza o pico existente
            if last_bar['high'] > last_confirmed_pivot['preco']:
                last_confirmed_pivot['preco'] = last_bar['high']
                last_confirmed_pivot['idx'] = df.index[-1]
            else:
                # Inverteu para baixa → cria um VALE
                potential_pivot = {
                    'idx': df.index[-1],
                    'tipo': 'VALE',
                    'preco': last_bar['low']
                }
                if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                    confirmed_pivots.append(potential_pivot)
        else:  # último pivô é VALE
            # Movimento ainda de baixa: atualiza o vale existente
            if last_bar['low'] < last_confirmed_pivot['preco']:
                last_confirmed_pivot['preco'] = last_bar['low']
                last_confirmed_pivot['idx'] = df.index[-1]
            else:
                # Inverteu para alta → cria um PICO
                potential_pivot = {
                    'idx': df.index[-1],
                    'tipo': 'PICO',
                    'preco': last_bar['high']
                }
                if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                    confirmed_pivots.append(potential_pivot)

    return confirmed_pivots


def is_head_extreme(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_bars: int) -> bool:
    """Valida se a cabeça é extrema (máxima/mínima) numa janela em barras."""
    base_lookback = int(avg_pivot_dist_bars *
                        Config.HEAD_EXTREME_LOOKBACK_FACTOR)
    lookback_bars = max(base_lookback, getattr(
        Config, 'HEAD_EXTREME_LOOKBACK_MIN_BARS', 30))
    if lookback_bars <= 0:
        return True

    try:
        # Encontra a posição numérica (iloc) do pivô da cabeça
        head_loc = df.index.get_loc(head_pivot['idx'])

        # Define o início e o fim da janela de busca em posições numéricas
        start_loc = max(0, head_loc - lookback_bars)
        end_loc = min(len(df), head_loc + lookback_bars + 1)

        # Fatia o DataFrame usando as posições para criar a janela de contexto
        context_df = df.iloc[start_loc:end_loc]

        if context_df.empty:
            return True

        if head_pivot['tipo'] == 'PICO':
            return head_pivot['preco'] >= context_df['high'].max()
        else:  # VALE
            return head_pivot['preco'] <= context_df['low'].min()

    except KeyError:
        # A data do pivô não foi encontrada no índice do DataFrame
        return False


def check_rsi_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    """Detect RSI divergence between p1 and p3 according to pattern type.

    Updated: Only considers divergences starting in overbought/oversold zones
    defined in Config. Returns a simple boolean for backward compatibility.

    For strength classification, use `assess_rsi_divergence_strength`.
    """
    try:
        direction = 'bearish' if tipo_padrao in ('OCO',) else 'bullish'
        src = df['high'] if tipo_padrao == 'OCO' else df['low']
        div, strong = assess_rsi_divergence_strength(
            df, p1_idx, p3_idx, p1_price, p3_price, direction, src
        )
        return bool(div)
    except Exception:
        return False


def check_macd_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    """Detecta divergência do histograma do MACD entre p1 e p3."""
    try:
        macd = df.ta.macd(fast=getattr(Config, 'MACD_FAST', 12), slow=getattr(
            Config, 'MACD_SLOW', 26), signal=getattr(Config, 'MACD_SIGNAL', 9), append=False)
        hist_col = 'MACDh_12_26_9'
        if p1_idx not in macd.index or p3_idx not in macd.index:
            return False
        hist_p1, hist_p3 = macd[hist_col].loc[p1_idx], macd[hist_col].loc[p3_idx]
        if tipo_padrao == 'OCO':
            return p3_price > p1_price and hist_p3 < hist_p1
        elif tipo_padrao == 'OCOI':
            return p3_price < p1_price and hist_p3 > hist_p1
    except Exception:
        return False
    return False

# --- New modular indicator helpers (Epic 1 RF-001..RF-004) ---


def assess_rsi_divergence_strength(
    df: pd.DataFrame,
    p1_idx,
    p3_idx,
    p1_price: float,
    p3_price: float,
    direction: str,
    source_series: pd.Series,
) -> Tuple[bool, bool]:
    """Assess RSI divergence between p1 and p3 with threshold gating and strength.

    - direction: 'bearish' or 'bullish'
    - source_series: price series used to compute RSI (e.g., df['close']|df['high']|df['low'])

    Returns (has_divergence, is_strong).
    Strong divergence requires deeper zone levels or a minimum delta in RSI.
    """
    try:
        rsi = ta.rsi(source_series, length=getattr(Config, 'RSI_LENGTH', 14))
        if p1_idx not in rsi.index or p3_idx not in rsi.index:
            return False, False
        rsi1, rsi3 = float(rsi.loc[p1_idx]), float(rsi.loc[p3_idx])
        if np.isnan(rsi1) or np.isnan(rsi3):
            return False, False

        overbought = getattr(Config, 'RSI_OVERBOUGHT', 70)
        oversold = getattr(Config, 'RSI_OVERSOLD', 30)
        strong_ob = getattr(Config, 'RSI_STRONG_OVERBOUGHT', 80)
        strong_os = getattr(Config, 'RSI_STRONG_OVERSOLD', 20)
        min_delta = getattr(Config, 'RSI_DIVERGENCE_MIN_DELTA', 5.0)

        if direction == 'bearish':
            # Price higher high, RSI lower high, starting from overbought
            valid_start = rsi1 >= overbought
            div = (p3_price > p1_price) and (rsi3 < rsi1) and valid_start
            strong = div and (rsi1 >= strong_ob or (rsi1 - rsi3) >= min_delta)
            return div, strong
        else:  # bullish
            # Price lower low, RSI higher low, starting from oversold
            valid_start = rsi1 <= oversold
            div = (p3_price < p1_price) and (rsi3 > rsi1) and valid_start
            strong = div and (rsi1 <= strong_os or (rsi3 - rsi1) >= min_delta)
            return div, strong
    except Exception:
        return False, False


def detect_macd_signal_cross(
    df: pd.DataFrame,
    idx_ref,
    direction: str,
    lookback_bars: Optional[int] = None,
) -> bool:
    """Detect MACD line crossing the signal line near a reference index.

    - direction: 'bearish' (MACD crosses below signal) or 'bullish' (above)
    - lookback_bars: how many bars back from idx_ref we allow the cross
    """
    try:
        lookback = lookback_bars if lookback_bars is not None else getattr(
            Config, 'MACD_SIGNAL_CROSS_LOOKBACK_BARS', 7
        )
        macd_df = df.ta.macd(fast=getattr(Config, 'MACD_FAST', 12), slow=getattr(
            Config, 'MACD_SLOW', 26), signal=getattr(Config, 'MACD_SIGNAL', 9), append=False)
        if macd_df is None:
            return False
        macd_line = macd_df.get('MACD_12_26_9')
        signal_line = macd_df.get('MACDs_12_26_9')
        if macd_line is None or signal_line is None:
            return False
        if idx_ref not in macd_df.index:
            return False
        ref_pos = df.index.get_loc(idx_ref)
        start_pos = max(0, ref_pos - lookback)
        # Use df index to slice consistent window
        window = df.index[start_pos:ref_pos + 1]
        diff = (macd_line.loc[window] - signal_line.loc[window]).dropna()
        if len(diff) < 2:
            return False
        prev, curr = diff.iloc[-2], diff.iloc[-1]
        if direction == 'bearish':
            return prev >= 0 and curr < 0
        return prev <= 0 and curr > 0
    except Exception:
        return False


def find_breakout_index(
    df: pd.DataFrame,
    neckline_price: float,
    start_idx,
    direction: str,
    max_bars: Optional[int] = None,
):
    """Find the first bar after start_idx where close breaks the neckline.

    - direction: 'bearish' (close < neckline) or 'bullish' (close > neckline)
    - Returns the index label of the breakout bar or None.
    """
    try:
        limit = max_bars if max_bars is not None else getattr(
            Config, 'BREAKOUT_SEARCH_MAX_BARS', 60
        )
        if start_idx not in df.index:
            return None
        start_pos = df.index.get_loc(start_idx)
        end_pos = min(len(df) - 1, start_pos + limit)
        for pos in range(start_pos + 1, end_pos + 1):
            idx = df.index[pos]
            close_val = float(df.loc[idx, 'close'])
            if direction == 'bearish' and close_val < neckline_price:
                return idx
            if direction == 'bullish' and close_val > neckline_price:
                return idx
        return None
    except Exception:
        return None


def check_breakout_volume(
    df: pd.DataFrame,
    breakout_idx,
    lookback_bars: Optional[int] = None,
    multiplier: Optional[float] = None,
) -> bool:
    """Confirm expressive volume increase on the breakout candle.

    Compares breakout volume vs. average volume of previous N bars.
    """
    try:
        if breakout_idx not in df.index:
            return False
        lookback = lookback_bars if lookback_bars is not None else getattr(
            Config, 'VOLUME_BREAKOUT_LOOKBACK_BARS', 20
        )
        mult = multiplier if multiplier is not None else getattr(
            Config, 'VOLUME_BREAKOUT_MULTIPLIER', 1.8
        )
        pos = df.index.get_loc(breakout_idx)
        start = max(0, pos - lookback)
        base = df.iloc[start:pos]['volume']
        if base.empty:
            return False
        base_mean = float(base.mean())
        breakout_vol = float(df.loc[breakout_idx, 'volume'])
        return base_mean > 0 and breakout_vol >= (mult * base_mean)
    except Exception:
        return False


def check_stochastic_confirmation(
    df: pd.DataFrame,
    p1_idx,
    p3_idx,
    p1_price: float,
    p3_price: float,
    direction: str,
) -> dict:
    """Stochastic Oscillator confirmations: divergence and %K/%D cross.

    Only considered if the starting value (at p1) is in overbought/oversold
    zones (>80 or <20). Returns a dict with two boolean flags:
    - valid_estocastico_divergencia
    - valid_estocastico_cross
    """
    result = {
        'valid_estocastico_divergencia': False,
        'valid_estocastico_cross': False,
    }
    try:
        stoch = ta.stoch(
            high=df['high'], low=df['low'], close=df['close'],
            k=getattr(Config, 'STOCH_K', 14),
            d=getattr(Config, 'STOCH_D', 3),
            smooth_k=getattr(Config, 'STOCH_SMOOTH_K', 3),
        )
        if stoch is None:
            return result
        k_col = f"STOCHk_{getattr(Config, 'STOCH_K', 14)}_{getattr(Config, 'STOCH_D', 3)}_{getattr(Config, 'STOCH_SMOOTH_K', 3)}"
        d_col = f"STOCHd_{getattr(Config, 'STOCH_K', 14)}_{getattr(Config, 'STOCH_D', 3)}_{getattr(Config, 'STOCH_SMOOTH_K', 3)}"
        if k_col not in stoch.columns or d_col not in stoch.columns:
            # Some pandas_ta versions name columns without underscores; try defaults
            k_col = 'STOCHk_14_3_3' if 'STOCHk_14_3_3' in stoch.columns else k_col
            d_col = 'STOCHd_14_3_3' if 'STOCHd_14_3_3' in stoch.columns else d_col
        if p1_idx not in stoch.index or p3_idx not in stoch.index:
            return result
        k1 = float(stoch.loc[p1_idx, k_col])
        k3 = float(stoch.loc[p3_idx, k_col])
        if np.isnan(k1) or np.isnan(k3):
            return result

        ob = getattr(Config, 'STOCH_OVERBOUGHT', 80)
        os_ = getattr(Config, 'STOCH_OVERSOLD', 20)
        min_delta = getattr(Config, 'STOCH_DIVERGENCE_MIN_DELTA', 5.0)

        # Divergence only if starting in OB/OS
        if direction == 'bearish' and k1 >= ob:
            div = (p3_price > p1_price) and (
                k3 < k1) and ((k1 - k3) >= min_delta)
            result['valid_estocastico_divergencia'] = bool(div)
        if direction == 'bullish' and k1 <= os_:
            div = (p3_price < p1_price) and (
                k3 > k1) and ((k3 - k1) >= min_delta)
            result['valid_estocastico_divergencia'] = bool(div)

        # Directional %K/%D cross within recent window
        if p3_idx in stoch.index:
            lookback = getattr(Config, 'STOCH_CROSS_LOOKBACK_BARS', 7)
            ref_pos = df.index.get_loc(p3_idx) if p3_idx in df.index else None
            if ref_pos is not None:
                start_pos = max(0, ref_pos - lookback)
                window = df.index[start_pos:ref_pos + 1]
                k = stoch.loc[window, k_col].dropna()
                d = stoch.loc[window, d_col].dropna()
                diff = (k - d).dropna()
                if len(diff) >= 2:
                    prev, curr = diff.iloc[-2], diff.iloc[-1]
                    if direction == 'bearish' and prev >= 0 and curr < 0 and k1 >= ob:
                        result['valid_estocastico_cross'] = True
                    if direction == 'bullish' and prev <= 0 and curr > 0 and k1 <= os_:
                        result['valid_estocastico_cross'] = True
    except Exception:
        return result
    return result


def check_volume_profile(df: pd.DataFrame, pivots: List[Dict[str, Any]], p1_idx, p3_idx, p5_idx) -> bool:
    """Compara volume próximo à cabeça vs. ombro direito para confirmar padrão."""
    try:
        indices = {p['idx']: i for i, p in enumerate(pivots)}
        idx_p1, idx_p3, idx_p5 = indices.get(
            p1_idx), indices.get(p3_idx), indices.get(p5_idx)
        if any(i is None for i in [idx_p1, idx_p3, idx_p5]) or idx_p1 < 2:
            return False
        p0_idx, p2_idx, p4_idx = pivots[idx_p1 -
                                        1]['idx'], pivots[idx_p3-1]['idx'], pivots[idx_p5-1]['idx']
        vol_cabeca = df.loc[p2_idx:p3_idx]['volume'].mean()
        vol_od = df.loc[p4_idx:p5_idx]['volume'].mean()
        return vol_cabeca > vol_od
    except Exception:
        return False
    return False


def validate_and_score_hns_pattern(p0, p1, p2, p3, p4, p5, p6, tipo_padrao, df_historico, pivots, avg_pivot_dist_bars):
    """Valida e pontua OCO/OCOI aplicando regras eliminatórias e confirmatórias."""
    details = {key: False for key in Config.SCORE_WEIGHTS_HNS.keys()}
    ombro_esq, neckline1, cabeca, neckline2, ombro_dir = p1, p2, p3, p4, p5

    # --- Lógica de validação original (sem alterações) ---
    altura_cabeca = abs(
        cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_esq = abs(
        ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_dir = abs(
        ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))

    details['valid_extremo_cabeca'] = (tipo_padrao == 'OCO' and cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']) or \
                                      (tipo_padrao == 'OCOI' and cabeca['preco'] <
                                       ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco'])
    if not details['valid_extremo_cabeca']:
        return None

    details['valid_contexto_cabeca'] = is_head_extreme(
        df_historico, cabeca, avg_pivot_dist_bars)
    if not details['valid_contexto_cabeca']:
        return None

    details['valid_simetria_ombros'] = altura_cabeca > 0 and \
        abs(altura_ombro_esq - altura_ombro_dir) <= altura_cabeca * \
        Config.SHOULDER_SYMMETRY_TOLERANCE
    if not details['valid_simetria_ombros']:
        return None

    details['valid_neckline_plana'] = altura_ombro_esq > 0 and \
        abs(neckline1['preco'] - neckline2['preco']
            ) <= altura_ombro_esq * Config.NECKLINE_FLATNESS_TOLERANCE
    if not details['valid_neckline_plana']:
        return None

    details['valid_base_tendencia'] = (tipo_padrao == 'OCO' and ((p0['preco'] < neckline1['preco'] and p0['preco'] < neckline2['preco']) or (abs(p0['preco'] - neckline1['preco']) < p0['preco'] * 0.05 and abs(p0['preco'] - neckline2['preco']) < p0['preco'] * 0.05))) or \
                                      (tipo_padrao == 'OCOI' and ((p0['preco'] > neckline1['preco'] and p0['preco'] > neckline2['preco']) or (abs(
                                          p0['preco'] - neckline1['preco']) < p0['preco'] * 0.05 and abs(p0['preco'] - neckline2['preco']) < p0['preco'] * 0.05)))
    if not details['valid_base_tendencia']:
        return None

    # reteste de neckline (p6) deve ocorrer próximo à neckline com tolerância por ATR
    neckline_price = np.mean([neckline1['preco'], neckline2['preco']])
    # Tolerância adaptativa baseada no ATR(14)
    atr_series = df_historico.ta.atr(length=14)
    # Procura o ATR no índice do p6; se não houver valor, usa o último ATR disponível
    if p6['idx'] in atr_series.index and not np.isnan(atr_series.loc[p6['idx']]):
        atr_val = atr_series.loc[p6['idx']]
    else:
        atr_val = atr_series.dropna(
        ).iloc[-1] if not atr_series.dropna().empty else 0

    max_variation = Config.NECKLINE_RETEST_ATR_MULTIPLIER * atr_val

    is_close_to_neckline = abs(p6['preco'] - neckline_price) <= max_variation
    details['valid_neckline_retest_p6'] = is_close_to_neckline
    if not details['valid_neckline_retest_p6']:
        return None

    # --- New indicator confirmations (Epic 1) ---
    direction = 'bearish' if tipo_padrao == 'OCO' else 'bullish'

    # RSI divergence with strength classification (threshold-gated)
    rsi_src = df_historico['high'] if tipo_padrao == 'OCO' else df_historico['low']
    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], direction, rsi_src
    )
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    # MACD histogram divergence (as before)
    if check_macd_divergence(df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], tipo_padrao):
        details['valid_divergencia_macd'] = True

    # Stochastic confirmations (divergence + %K/%D cross, zone-gated)
    stoch_flags = check_stochastic_confirmation(
        df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], direction
    )
    details.update(stoch_flags)

    # Breakout detection from neckline between p5 -> next bars
    breakout_idx = find_breakout_index(
        df_historico, neckline_price, p5['idx'], direction, getattr(
            Config, 'BREAKOUT_SEARCH_MAX_BARS', 60)
    )

    # MACD signal cross near breakout
    if breakout_idx is not None and detect_macd_signal_cross(df_historico, breakout_idx, direction):
        details['valid_macd_signal_cross'] = True
    elif detect_macd_signal_cross(df_historico, p6['idx'], direction):
        # fallback: around retest
        details['valid_macd_signal_cross'] = True

    # Volume spike on the breakout candle
    if breakout_idx is not None and check_breakout_volume(df_historico, breakout_idx):
        details['valid_volume_breakout_neckline'] = True

    # Existing optional checks
    if altura_ombro_esq > 0 and (altura_cabeca / altura_ombro_esq >= Config.HEAD_SIGNIFICANCE_RATIO) and (altura_cabeca / altura_ombro_dir >= Config.HEAD_SIGNIFICANCE_RATIO):
        details['valid_proeminencia_cabeca'] = True
    if check_volume_profile(df_historico, pivots, p1['idx'], p3['idx'], p5['idx']):
        details['valid_perfil_volume'] = True
    if (tipo_padrao == 'OCO' and ombro_dir['preco'] < ombro_esq['preco']) or \
       (tipo_padrao == 'OCOI' and ombro_dir['preco'] > ombro_esq['preco']):
        details['valid_ombro_direito_fraco'] = True

    # Soma de pontuação de confirmações
    score = 0
    for rule, passed in details.items():
        if passed:
            score += Config.SCORE_WEIGHTS_HNS.get(rule, 0)

    if score >= Config.MINIMUM_SCORE_HNS:
        base_data = {
            'padrao_tipo': tipo_padrao, 'score_total': score,
            'p0_idx': p0['idx'],
            'ombro1_idx': p1['idx'], 'ombro1_preco': p1['preco'],
            'neckline1_idx': p2['idx'], 'neckline1_preco': p2['preco'],
            'cabeca_idx': p3['idx'], 'cabeca_preco': p3['preco'],
            'neckline2_idx': p4['idx'], 'neckline2_preco': p4['preco'],
            'ombro2_idx': p5['idx'], 'ombro2_preco': p5['preco'],
            'retest_p6_idx': p6['idx'], 'retest_p6_preco': p6['preco']
        }
        base_data.update(details)
        return base_data

    return None


# --- FUNÇÕES DE VALIDAÇÃO OPCIONAL (DT/DB) ---

def check_volume_profile_dtb(df, p0, p1, p2, p3):
    try:
        idx0, idx1, idx2, idx3 = p0.get('idx'), p1.get(
            'idx'), p2.get('idx'), p3.get('idx')
        if any(idx not in df.index for idx in [idx0, idx1, idx2, idx3]):
            return False

        # Ensure ascending intervals
        start1, end1 = (idx0, idx1) if idx0 <= idx1 else (idx1, idx0)
        start2, end2 = (idx2, idx3) if idx2 <= idx3 else (idx3, idx2)

        vol_extremo_1 = df.loc[start1:end1]['volume'].mean()
        vol_extremo_2 = df.loc[start2:end2]['volume'].mean()

        if np.isnan(vol_extremo_1) or np.isnan(vol_extremo_2):
            return False

        # Expect decreasing volume into the second extreme
        return vol_extremo_2 < vol_extremo_1
    except Exception:
        return False


def check_obv_divergence_dtb(df, p1, p3, tipo_padrao):
    try:
        # Gera/garante coluna OBV
        df.ta.obv(append=True)
        if 'OBV' not in df.columns:
            return False

        idx1, idx3 = p1.get('idx'), p3.get('idx')
        if idx1 not in df.index or idx3 not in df.index:
            return False

        obv_p1 = df.loc[idx1, 'OBV']
        obv_p3 = df.loc[idx3, 'OBV']
        if np.isnan(obv_p1) or np.isnan(obv_p3):
            return False

        if tipo_padrao == 'DT':
            return obv_p3 < obv_p1
        if tipo_padrao == 'DB':
            return obv_p3 > obv_p1
    except Exception:
        return False
    return False


def identificar_padroes_hns(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    """Gera janelas de 7 pivôs, identifica OCO/OCOI e valida com p6 (reteste)."""
    padroes_encontrados = []
    n = len(pivots)
    if n < 7:
        return []
    try:
        # Cria um mapeador de timestamp para posição numérica para eficiência
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        distancias_em_barras = [
            locs[pivots[i]['idx']] - locs[pivots[i-1]['idx']]
            for i in range(1, n)
            if pivots[i]['idx'] in locs and pivots[i-1]['idx'] in locs
        ]
        avg_pivot_dist_bars = np.mean(
            distancias_em_barras) if distancias_em_barras else 0
    except Exception as e:
        print(f"{Fore.YELLOW}Aviso: Não foi possível calcular a distância média dos pivôs. Erro: {e}{Style.RESET_ALL}")
        avg_pivot_dist_bars = 0  # Fallback
    start_index = max(0, n - 6 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)

    print(
        f"Analyzing only the last {Config.RECENT_PATTERNS_LOOKBACK_COUNT} possible final pivots (from index {start_index}).")

    for i in range(start_index, n - 6):
        janela = pivots[i:i+7]
        p0, p1, p2, p3, p4, p5 = janela[0], janela[1], janela[2], janela[3], janela[4], janela[5]
        p6 = janela[6]

        tipo_padrao = None
        # Check H&S (p6 is the retest)
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'OCO'
        # Check Inverse H&S
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'OCOI'

        if tipo_padrao:
            dados_padrao = validate_and_score_hns_pattern(
                p0, p1, p2, p3, p4, p5, p6, tipo_padrao, df_historico, pivots, avg_pivot_dist_bars)
            if dados_padrao:
                padroes_encontrados.append(dados_padrao)

    return padroes_encontrados


def validate_and_score_double_pattern(p0, p1, p2, p3, p4, tipo_padrao, df_historico, avg_pivot_dist_bars: int):
    """Validate and score Double Top (DT) and Double Bottom (DB).

    p4: pivô adicional (reteste).
    """
    if tipo_padrao not in ('DT', 'DB'):
        return None

    details = {key: False for key in Config.SCORE_WEIGHTS_DTB.keys()}
    debug = getattr(Config, 'DTB_DEBUG', False)

    preco_p0, preco_p1 = float(p0['preco']), float(p1['preco'])
    preco_p2, preco_p3 = float(p2['preco']), float(p3['preco'])

    # Structure: expected pivot types and basic price relations
    if tipo_padrao == 'DT':
        estrutura_tipos_ok = (
            p1.get('tipo') == 'PICO' and p2.get(
                'tipo') == 'VALE' and p3.get('tipo') == 'PICO'
        )
        relacoes_precos_ok = (preco_p1 > preco_p0) and (
            preco_p1 > preco_p2 and preco_p3 > preco_p2)
    else:  # 'DB'
        estrutura_tipos_ok = (
            p1.get('tipo') == 'VALE' and p2.get(
                'tipo') == 'PICO' and p3.get('tipo') == 'VALE'
        )
        relacoes_precos_ok = (preco_p1 < preco_p0) and (
            preco_p1 < preco_p2 and preco_p3 < preco_p2)

    details['valid_estrutura_picos_vales'] = estrutura_tipos_ok and relacoes_precos_ok
    if not details['valid_estrutura_picos_vales']:
        if debug:
            _dtb_debug(
                f"{Fore.YELLOW}DTB debug: fail at valid_estrutura_picos_vales ({tipo_padrao}). "
                f"tipos=[{p0.get('tipo')},{p1.get('tipo')},{p2.get('tipo')},{p3.get('tipo')}] "
                f"precos=[p0={preco_p0:.6f}, p1={preco_p1:.6f}, p2={preco_p2:.6f}, p3={preco_p3:.6f}]"
                f"{Style.RESET_ALL}")
        return None

    # Contexto obrigatório: p1 e p3 devem ser extremos relevantes na janela
    try:
        p1_context_ok = is_head_extreme(df_historico, p1, avg_pivot_dist_bars)
        # p3_context_ok = is_head_extreme(df_historico, p3, avg_pivot_dist_bars)
    except Exception:
        p1_context_ok = False
    details['valid_contexto_extremos'] = bool(p1_context_ok)
    if not details['valid_contexto_extremos']:
        if debug:
            # Reproduz minimamente a janela de contexto para depuração
            try:
                base_lookback = int(avg_pivot_dist_bars *
                                    Config.HEAD_EXTREME_LOOKBACK_FACTOR)
                lookback_bars = max(base_lookback, getattr(
                    Config, 'HEAD_EXTREME_LOOKBACK_MIN_BARS', 30))
                head_loc = df_historico.index.get_loc(p1['idx'])
                start_loc = max(0, head_loc - lookback_bars)
                end_loc = min(len(df_historico), head_loc + lookback_bars + 1)
                context_df = df_historico.iloc[start_loc:end_loc]
                ctx_high = context_df['high'].max(
                ) if not context_df.empty else float('nan')
                ctx_low = context_df['low'].min(
                ) if not context_df.empty else float('nan')
                _dtb_debug(
                    f"{Fore.YELLOW}DTB debug: fail at valid_contexto_extremos ({tipo_padrao}). "
                    f"lookback_bars={lookback_bars} p1_preco={preco_p1:.6f} ctx_high={ctx_high:.6f} ctx_low={ctx_low:.6f}{Style.RESET_ALL}")
            except Exception:
                _dtb_debug(
                    f"{Fore.YELLOW}DTB debug: fail at valid_contexto_extremos ({tipo_padrao}). "
                    f"[context window calc error]{Style.RESET_ALL}")
        return None

    # Trend context: enforce HH/HL for DT and LH/LL for DB on recent pivots
    try:
        # HH/HL or LH/LL with minimum separation to avoid noise
        min_sep = Config.DTB_TREND_MIN_DIFF_FACTOR * \
            max(1.0, abs(preco_p1 - preco_p2))
        if tipo_padrao == 'DT':
            # higher low: p2 >= p0 within tolerance
            hl_ok = (preco_p2 >= preco_p0 - min_sep)
            trend_ok = hl_ok
        else:  # DB
            # lower high: p2 <= p0 within tolerance
            lh_ok = (preco_p2 <= preco_p0 + min_sep)
            trend_ok = lh_ok
    except Exception:
        trend_ok = False
    details['valid_contexto_tendencia'] = bool(trend_ok)
    if not details['valid_contexto_tendencia']:
        if debug:
            _dtb_debug(
                f"{Fore.YELLOW}DTB debug: fail at valid_contexto_tendencia ({tipo_padrao}). "
                f"p0={preco_p0:.6f} p1={preco_p1:.6f} p2={preco_p2:.6f} min_sep={min_sep:.9f}{Style.RESET_ALL}")
        return None

    # Symmetry of extremes (p1 ~ p3) based on pattern height (|p1 - p2|)
    altura_padrao = abs(preco_p1 - preco_p2)
    tolerancia_preco = Config.DTB_SYMMETRY_TOLERANCE_FACTOR * altura_padrao
    diff_picos = abs(preco_p1 - preco_p3)
    details['valid_simetria_extremos'] = diff_picos <= tolerancia_preco
    if not details['valid_simetria_extremos']:
        if debug:
            _dtb_debug(
                f"{Fore.YELLOW}DTB debug: fail at valid_simetria_extremos ({tipo_padrao}). "
                f"tol={tolerancia_preco:.9f} diff={diff_picos:.9f} altura={altura_padrao:.9f} "
                f"p1={preco_p1:.6f} p3={preco_p3:.6f}{Style.RESET_ALL}")
        return None

    # Depth of middle valley/peak relative to previous leg (p0->p1)
    perna_anterior = abs(preco_p1 - preco_p0)
    if tipo_padrao == 'DT':
        profundidade = preco_p1 - preco_p2
    else:  # 'DB'
        profundidade = preco_p2 - preco_p1
    required = Config.DTB_VALLEY_PEAK_DEPTH_RATIO * perna_anterior
    details['valid_profundidade_vale_pico'] = perna_anterior > 0 and profundidade >= required
    if not details['valid_profundidade_vale_pico']:
        if debug:
            _dtb_debug(
                f"{Fore.YELLOW}DTB debug: fail at valid_profundidade_vale_pico ({tipo_padrao}). "
                f"profundidade={profundidade:.6f} required={required:.6f} perna_anterior={perna_anterior:.6f} "
                f"ratio_req={Config.DTB_VALLEY_PEAK_DEPTH_RATIO:.3f}{Style.RESET_ALL}")
        return None

    # Mandatory: p4 must be a valid retest of the neckline (defined by p2)
    neckline_price = preco_p2
    atr_series = df_historico.ta.atr(length=14)
    if p4.get('idx') in atr_series.index and not np.isnan(atr_series.loc[p4.get('idx')]):
        atr_val = atr_series.loc[p4.get('idx')]
    else:
        atr_val = atr_series.dropna(
        ).iloc[-1] if not atr_series.dropna().empty else 0

    max_variation = Config.NECKLINE_RETEST_ATR_MULTIPLIER * atr_val
    dist_neck = abs(float(p4.get('preco')) - neckline_price)
    inside_tolerance = dist_neck <= max_variation
    details['valid_neckline_retest_p4'] = inside_tolerance
    if not details['valid_neckline_retest_p4']:
        if debug:
            _dtb_debug(
                f"{Fore.YELLOW}DTB debug: fail at valid_neckline_retest_p4 ({tipo_padrao}). "
                f"inside_tol={inside_tolerance} atr={atr_val:.6f} mult={Config.NECKLINE_RETEST_ATR_MULTIPLIER:.3f} "
                f"atr*mult={max_variation:.6f} neckline={neckline_price:.6f} p4={float(p4.get('preco')):.6f} "
                f"dist={dist_neck:.6f}{Style.RESET_ALL}")
        return None

    # Optional confirmations (volume profile and divergences)
    details['valid_perfil_volume_decrescente'] = check_volume_profile_dtb(
        df_historico, p0, p1, p2, p3
    )
    details['valid_divergencia_obv'] = check_obv_divergence_dtb(
        df_historico, p1, p3, tipo_padrao
    )

    # New modular indicator confirmations (Epic 1)
    direction = 'bearish' if tipo_padrao == 'DT' else 'bullish'

    # RSI divergence with strength (use close for DT/DB)
    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1.get('idx'), p3.get(
            'idx'), preco_p1, preco_p3, direction, df_historico['close']
    )
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    # MACD signal cross used as confirmation for DT/DB (legacy divergence removed)

    stoch_flags = check_stochastic_confirmation(
        df_historico, p1.get('idx'), p3.get(
            'idx'), preco_p1, preco_p3, direction
    )
    details.update(stoch_flags)

    # Breakout index: from p3 crossing neckline (p2)
    breakout_idx = find_breakout_index(
        df_historico, neckline_price, p3.get('idx'), direction, getattr(
            Config, 'BREAKOUT_SEARCH_MAX_BARS', 60)
    )

    # MACD signal cross near breakout
    if breakout_idx is not None and detect_macd_signal_cross(df_historico, breakout_idx, direction):
        details['valid_macd_signal_cross'] = True
        details['valid_divergencia_macd'] = True
    elif detect_macd_signal_cross(df_historico, p4.get('idx'), direction):
        details['valid_macd_signal_cross'] = True
        details['valid_divergencia_macd'] = True

    # Volume spike on breakout
    if breakout_idx is not None and check_breakout_volume(df_historico, breakout_idx):
        details['valid_volume_breakout_neckline'] = True

    # Optional: second top lower (DT) or second bottom higher (DB)
    if 'valid_segundo_topo_menor' in details:
        if tipo_padrao == 'DT' and preco_p3 < preco_p1:
            details['valid_segundo_topo_menor'] = True
        elif tipo_padrao == 'DB' and preco_p3 > preco_p1:
            details['valid_segundo_topo_menor'] = True

    # Scoring
    score = 0
    for rule, passed in details.items():
        if passed:
            score += Config.SCORE_WEIGHTS_DTB.get(rule, 0)

    if score >= Config.MINIMUM_SCORE_DTB:
        if debug:
            _dtb_debug(
                f"{Fore.GREEN}DTB debug: ACCEPTED {tipo_padrao} with score={score}{Style.RESET_ALL}")
        return {
            'padrao_tipo': tipo_padrao,
            'score_total': score,
            'p0_idx': p0['idx'], 'p0_preco': preco_p0,
            'p1_idx': p1['idx'], 'p1_preco': preco_p1,
            'p2_idx': p2['idx'], 'p2_preco': preco_p2,
            'p3_idx': p3['idx'], 'p3_preco': preco_p3,
            'p4_idx': p4['idx'], 'p4_preco': float(p4['preco']),
            **details
        }

    if debug:
        _dtb_debug(
            f"{Fore.YELLOW}DTB debug: fail at minimum score ({tipo_padrao}). score={score} min={Config.MINIMUM_SCORE_DTB}{Style.RESET_ALL}")
    return None


def identificar_padroes_double_top_bottom(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    """Slide 5-pivot windows (with retest) and validate DT/DB with symmetry/depth rules."""
    padroes_encontrados: List[Dict[str, Any]] = []
    n = len(pivots)
    if n < 5:
        return []

    # Calcula distância média entre pivôs (em barras) para definir janela de contexto
    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        distancias_em_barras = [
            locs[pivots[i]['idx']] - locs[pivots[i-1]['idx']]
            for i in range(1, n)
            if pivots[i]['idx'] in locs and pivots[i-1]['idx'] in locs
        ]
        avg_pivot_dist_bars = np.mean(
            distancias_em_barras) if distancias_em_barras else 0
    except Exception as e:
        print(f"{Fore.YELLOW}Aviso: Não foi possível calcular a distância média dos pivôs (DTB). Erro: {e}{Style.RESET_ALL}")
        avg_pivot_dist_bars = 0

    start_index = max(0, n - 4 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)
    print(
        f"Analyzing only the last {Config.RECENT_PATTERNS_LOOKBACK_COUNT} DT/DB candidates (from index {start_index}).")

    for i in range(start_index, n - 4):
        janela = pivots[i:i+5]
        p0, p1, p2, p3 = janela[0], janela[1], janela[2], janela[3]
        p4 = janela[4]

        tipo_padrao = None
        # Verificação base: usa os 4 primeiros pivôs (p0..p3)
        if all(p.get('tipo') == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'DT'
        elif all(p.get('tipo') == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'DB'

        if tipo_padrao:
            dados_padrao = validate_and_score_double_pattern(
                p0, p1, p2, p3, p4, tipo_padrao, df_historico, avg_pivot_dist_bars)
            if dados_padrao:
                padroes_encontrados.append(dados_padrao)

    return padroes_encontrados


# --- Triple Top/Bottom (TT/TB) detection & validation (standalone; not wired in main) ---

def identificar_padroes_ttb(pivots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect Triple Top (TT) and Triple Bottom (TB) 7-pivot sequences.

    Pattern windows (types):
    - TT: ['VALE','PICO','VALE','PICO','VALE','PICO','VALE']
    - TB: ['PICO','VALE','PICO','VALE','PICO','VALE','PICO']

    Returns a list of pattern dicts with keys: 'padrao_tipo' and p0..p6 as 'p{k}_obj'.
    """
    resultados: List[Dict[str, Any]] = []
    n = len(pivots)
    if n < 7:
        return resultados

    for i in range(0, n - 6):
        janela = pivots[i:i + 7]
        tipos = [p.get('tipo') for p in janela]
        if tipos == ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE']:
            resultados.append(
                {'padrao_tipo': 'TT', **{f'p{k}_obj': janela[k] for k in range(7)}})
        elif tipos == ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO']:
            resultados.append(
                {'padrao_tipo': 'TB', **{f'p{k}_obj': janela[k] for k in range(7)}})
    return resultados


def validate_and_score_triple_pattern(pattern: Dict[str, Any], df_historico: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Validate and score Triple Top (TT) or Triple Bottom (TB) using DTB confirmations.

    Reuses DTB confirmation helpers (RSI, MACD divergence and signal cross, Stochastic, OBV,
    breakout volume). Returns a dict with 'padrao_tipo', 'score_total', p0..p6 *_idx/_preco and
    'valid_*' flags. Does not integrate into the main pipeline.
    """
    tipo = pattern.get('padrao_tipo')
    if tipo not in ('TT', 'TB'):
        return None

    # Unpack pivots
    pivs = [pattern.get(f'p{k}_obj') for k in range(7)]
    if any(p is None for p in pivs):
        return None
    p0, p1, p2, p3, p4, p5, p6 = pivs

    # Prices
    preco_p0 = float(p0['preco'])
    preco_p1 = float(p1['preco'])
    preco_p2 = float(p2['preco'])
    preco_p3 = float(p3['preco'])
    preco_p4 = float(p4['preco'])
    preco_p5 = float(p5['preco'])
    preco_p6 = float(p6['preco'])

    # Avg pivot distance (bars) for context
    try:
        locs = pd.Series(range(len(df_historico)), index=df_historico.index)
        dist_barras = [locs[pivs[i]['idx']] - locs[pivs[i-1]['idx']]
                       for i in range(1, 7)]
        avg_pivot_dist_bars = float(
            np.mean(dist_barras)) if dist_barras else 0.0
    except Exception:
        avg_pivot_dist_bars = 0.0

    details = {key: False for key in Config.SCORE_WEIGHTS_TTB.keys()}

    # Structure check
    if tipo == 'TT':
        estrutura_ok = (
            p1.get('tipo') == 'PICO' and p2.get('tipo') == 'VALE' and
            p3.get('tipo') == 'PICO' and p4.get('tipo') == 'VALE' and
            p5.get('tipo') == 'PICO'
        )
        rel_ok = (
            preco_p1 > preco_p0 and preco_p1 > preco_p2 and
            preco_p3 > preco_p2 and preco_p3 > preco_p4 and
            preco_p5 > preco_p4 and preco_p5 > preco_p6
        )
    else:  # TB
        estrutura_ok = (
            p1.get('tipo') == 'VALE' and p2.get('tipo') == 'PICO' and
            p3.get('tipo') == 'VALE' and p4.get('tipo') == 'PICO' and
            p5.get('tipo') == 'VALE'
        )
        rel_ok = (
            preco_p1 < preco_p0 and preco_p1 < preco_p2 and
            preco_p3 < preco_p2 and preco_p3 < preco_p4 and
            preco_p5 < preco_p4 and preco_p5 < preco_p6
        )
    details['valid_estrutura_picos_vales'] = bool(estrutura_ok and rel_ok)
    if not details['valid_estrutura_picos_vales']:
        return None

    # Context extremos (reuse head extreme logic)
    try:
        p1_ctx = is_head_extreme(df_historico, p1, avg_pivot_dist_bars)
    except Exception:
        p1_ctx = False
    details['valid_contexto_extremos'] = bool(p1_ctx)
    if not details['valid_contexto_extremos']:
        return None

    # Trend context (HL for TT, LH for TB) with tolerance
    try:
        min_sep = Config.DTB_TREND_MIN_DIFF_FACTOR * \
            max(1.0, abs(preco_p1 - preco_p2))
        if tipo == 'TT':
            hl1 = (preco_p2 >= preco_p0 - min_sep)
            hl2 = (preco_p4 >= preco_p2 - min_sep)
            trend_ok = hl1 and hl2
        else:
            lh1 = (preco_p2 <= preco_p0 + min_sep)
            lh2 = (preco_p4 <= preco_p2 + min_sep)
            trend_ok = lh1 and lh2
    except Exception:
        trend_ok = False
    details['valid_contexto_tendencia'] = bool(trend_ok)
    if not details['valid_contexto_tendencia']:
        return None

    # Symmetry across three extremes using tolerance vs pattern height
    try:
        tol = Config.DTB_SYMMETRY_TOLERANCE_FACTOR
        if tipo == 'TT':
            alturas = [abs(preco_p1 - preco_p2), abs(preco_p3 -
                                                     preco_p4), abs(preco_p5 - preco_p6)]
            altura_ref = np.mean(alturas) if alturas else 0.0
            peaks = [preco_p1, preco_p3, preco_p5]
            diff_span = max(peaks) - min(peaks)
            details['valid_simetria_extremos'] = altura_ref > 0 and diff_span <= tol * altura_ref
        else:
            alturas = [abs(preco_p2 - preco_p1), abs(preco_p4 -
                                                     preco_p3), abs(preco_p6 - preco_p5)]
            altura_ref = np.mean(alturas) if alturas else 0.0
            valleys = [preco_p1, preco_p3, preco_p5]
            diff_span = max(valleys) - min(valleys)
            details['valid_simetria_extremos'] = altura_ref > 0 and diff_span <= tol * altura_ref
    except Exception:
        details['valid_simetria_extremos'] = False
    if not details['valid_simetria_extremos']:
        return None

    # Depth checks for each leg after extreme
    try:
        perna_anterior = abs(preco_p1 - preco_p0)
        if tipo == 'TT':
            d1 = preco_p1 - preco_p2
            d2 = preco_p3 - preco_p4
            d3 = preco_p5 - preco_p6
        else:
            d1 = preco_p2 - preco_p1
            d2 = preco_p4 - preco_p3
            d3 = preco_p6 - preco_p5
        required = Config.DTB_VALLEY_PEAK_DEPTH_RATIO * perna_anterior
        details['valid_profundidade_vale_pico'] = perna_anterior > 0 and all(
            d >= required for d in [d1, d2, d3])
    except Exception:
        details['valid_profundidade_vale_pico'] = False
    if not details['valid_profundidade_vale_pico']:
        return None

    # Neckline retest at p6 relative to avg(p2, p4)
    neckline_price = float((preco_p2 + preco_p4) / 2.0)
    atr_series = df_historico.ta.atr(length=14)
    if p6.get('idx') in atr_series.index and not np.isnan(atr_series.loc[p6.get('idx')]):
        atr_val = float(atr_series.loc[p6.get('idx')])
    else:
        atr_val = float(
            atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0
    max_var = Config.NECKLINE_RETEST_ATR_MULTIPLIER * atr_val
    details['valid_neckline_retest_p6'] = abs(
        preco_p6 - neckline_price) <= max_var
    if not details['valid_neckline_retest_p6']:
        return None

    # Optional confirmations (reuse DTB helpers)
    details['valid_perfil_volume_decrescente'] = check_volume_profile_dtb(
        df_historico, p0, p1, p2, p3)

    details['valid_divergencia_obv'] = check_obv_divergence_dtb(
        df_historico, p1, p5, 'DT' if tipo == 'TT' else 'DB')

    direction = 'bearish' if tipo == 'TT' else 'bullish'
    rsi_div, rsi_strong = assess_rsi_divergence_strength(
        df_historico, p1.get('idx'), p5.get(
            'idx'), preco_p1, preco_p5, direction, df_historico['close']
    )
    if rsi_div:
        details['valid_divergencia_rsi'] = True
    if rsi_strong:
        details['valid_divergencia_rsi_strong'] = True

    st_flags = check_stochastic_confirmation(df_historico, p1.get(
        'idx'), p5.get('idx'), preco_p1, preco_p5, direction)
    details.update(st_flags)

    breakout_idx = find_breakout_index(df_historico, neckline_price, p6.get(
        'idx'), direction, getattr(Config, 'BREAKOUT_SEARCH_MAX_BARS', 60))
    if breakout_idx is not None and detect_macd_signal_cross(df_historico, breakout_idx, direction):
        details['valid_macd_signal_cross'] = True
        details['valid_divergencia_macd'] = True
    elif detect_macd_signal_cross(df_historico, p6.get('idx'), direction):
        details['valid_macd_signal_cross'] = True
        details['valid_divergencia_macd'] = True

    if breakout_idx is not None and check_breakout_volume(df_historico, breakout_idx):
        details['valid_volume_breakout_neckline'] = True

    # Scoring
    score = 0
    for rule, passed in details.items():
        if passed:
            score += Config.SCORE_WEIGHTS_TTB.get(rule, 0)

    if score >= Config.MINIMUM_SCORE_TTB:
        base = {
            'padrao_tipo': tipo,
            'score_total': score,
            'p0_idx': p0['idx'], 'p0_preco': preco_p0,
            'p1_idx': p1['idx'], 'p1_preco': preco_p1,
            'p2_idx': p2['idx'], 'p2_preco': preco_p2,
            'p3_idx': p3['idx'], 'p3_preco': preco_p3,
            'p4_idx': p4['idx'], 'p4_preco': preco_p4,
            'p5_idx': p5['idx'], 'p5_preco': preco_p5,
            'p6_idx': p6['idx'], 'p6_preco': preco_p6,
        }
        base.update(details)
        return base

    return None


def _parse_cli_args() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando do gerador."""
    parser = argparse.ArgumentParser(
        description="Gerador de dataset de padrões (OCO/OCOI, DT/DB)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Lista de tickers separados por vírgula (ex.: BTC-USD,ETH-USD). Default: Config.TICKERS",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Lista de estratégias ZigZag separadas por vírgula (ex.: swing_short,intraday_momentum). Default: todas",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default=None,
        help="Lista de intervalos separados por vírgula (ex.: 5m,15m,1h,4h,1d). Default: todos por estratégia",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="Período do yfinance (ex.: 5y, 2y, 7d). Default: Config.DATA_PERIOD",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho do CSV de saída. Default: Config.FINAL_CSV_PATH",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="ALL",
        help="Tipos de padrões a serem detectados, separados por vírgula (ex: HNS,DTB,TTB). Default: ALL",
    )
    return parser.parse_args()


def main():
    """Pipeline de geração: baixar dados, detectar padrões, salvar CSV final."""
    args = _parse_cli_args()

    # Filtros opcionais via CLI (mantendo defaults do Config)
    selected_tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else Config.TICKERS
    )

    if args.period:
        Config.DATA_PERIOD = args.period

    intervals_filter = (
        {i.strip() for i in args.intervals.split(",") if i.strip()}
        if args.intervals
        else None
    )

    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",") if s.strip()}
        strategies_dict = {
            name: cfg for name, cfg in Config.ZIGZAG_STRATEGIES.items() if name in wanted
        }
    else:
        strategies_dict = Config.ZIGZAG_STRATEGIES

    final_csv_path = args.output if args.output else Config.FINAL_CSV_PATH

    wanted_patterns = args.patterns.upper()

    print(f"{Style.BRIGHT}--- STARTING GENERATION ENGINE (v20 - Strategies) ---")
    os.makedirs(os.path.dirname(final_csv_path)
                or Config.OUTPUT_DIR, exist_ok=True)

    todos_os_padroes_finais = []

    for strategy_name, intervals_config in strategies_dict.items():
        print(f"\n{Style.BRIGHT}===== STRATEGY: {strategy_name.upper()} =====")
        for interval, params in intervals_config.items():
            if intervals_filter and interval not in intervals_filter:
                continue
            for ticker in selected_tickers:
                print(
                    f"\n--- Processing: {ticker} | Interval: {interval} (Strategy: {strategy_name}) ---")
                try:
                    df_historico = buscar_dados(
                        ticker, Config.DATA_PERIOD, interval)

                    print(
                        f"Calculando ZigZag com depth={params['depth']}, deviation={params['deviation']}%...")
                    pivots_detectados = calcular_zigzag_oficial(
                        df_historico, params['depth'], params['deviation'])

                    if len(pivots_detectados) < 4:
                        print("ℹ️ Not enough pivots to form a pattern.")
                        continue

                    todos_os_padroes_nesta_execucao: List[Dict[str, Any]] = []

                    if ('ALL' in wanted_patterns or 'HNS' in wanted_patterns) and len(pivots_detectados) >= 7:
                        print("Identifying H&S patterns with hard rules...")
                        padroes_hns_encontrados = identificar_padroes_hns(
                            pivots_detectados, df_historico)
                        todos_os_padroes_nesta_execucao.extend(
                            padroes_hns_encontrados)

                    if 'ALL' in wanted_patterns or 'DTB' in wanted_patterns:
                        padroes_dtb_encontrados = identificar_padroes_double_top_bottom(
                            pivots_detectados, df_historico)
                        todos_os_padroes_nesta_execucao.extend(
                            padroes_dtb_encontrados)

                    # Triple Top/Bottom integration (TT/TB)
                    if 'ALL' in wanted_patterns or 'TTB' in wanted_patterns:
                        print("Identifying Triple Top/Bottom (TT/TB) candidates...")
                        candidatos_ttb = identificar_padroes_ttb(
                            pivots_detectados)
                        if candidatos_ttb:
                            print(
                                f"Found {len(candidatos_ttb)} TT/TB raw candidates. Validating...")
                        for cand in candidatos_ttb:
                            dados_ttb = validate_and_score_triple_pattern(
                                cand, df_historico)
                            if dados_ttb:
                                # Enrich with metadata and optional columns (tipo/score)
                                dados_ttb['strategy'] = strategy_name
                                dados_ttb['timeframe'] = interval
                                dados_ttb['ticker'] = ticker
                                dados_ttb['tipo'] = dados_ttb.get(
                                    'padrao_tipo')
                                dados_ttb['score'] = dados_ttb.get(
                                    'score_total')
                                print(
                                    f"{Fore.GREEN}TTB accepted {dados_ttb['tipo']} with score={dados_ttb['score']}{Style.RESET_ALL}")
                                todos_os_padroes_nesta_execucao.append(
                                    dados_ttb)

                    if todos_os_padroes_nesta_execucao:
                        print(
                            f"{Fore.GREEN}✅ Found {len(todos_os_padroes_nesta_execucao)} H&S/DT/DB patterns passing rules and score.")
                        for padrao in todos_os_padroes_nesta_execucao:
                            padrao['strategy'] = strategy_name
                            padrao['timeframe'] = interval
                            padrao['ticker'] = ticker
                            todos_os_padroes_finais.append(padrao)
                    else:
                        print(
                            "ℹ️ No H&S or DT/DB patterns met the criteria or minimum score.")
                except Exception as e:
                    print(
                        f"{Fore.RED}❌ Error processing {ticker}/{interval} on strategy {strategy_name}: {e}")

    print(
        f"\n{Style.BRIGHT}--- Finished. Saving dataset... ---{Style.RESET_ALL}")

    if not todos_os_padroes_finais:
        print(
            f"{Fore.YELLOW}No H&S or DT/DB patterns were found.")
        return

    df_final = pd.DataFrame(todos_os_padroes_finais)

    # Build unique key per pattern based on last relevant pivot (datetime dtype)
    # For H&S use 'cabeca_idx'; for DT/DB use 'p3_idx'
    if 'cabeca_idx' not in df_final.columns:
        df_final['cabeca_idx'] = pd.NaT
    if 'p3_idx' not in df_final.columns:
        df_final['p3_idx'] = pd.NaT
    # First coerce to datetime, then strip timezone to avoid tz-aware vs naive mix
    df_final['cabeca_idx'] = pd.to_datetime(
        df_final['cabeca_idx'], errors='coerce')
    df_final['p3_idx'] = pd.to_datetime(df_final['p3_idx'], errors='coerce')
    try:
        df_final['cabeca_idx'] = df_final['cabeca_idx'].dt.tz_localize(None)
    except Exception:
        pass
    try:
        df_final['p3_idx'] = df_final['p3_idx'].dt.tz_localize(None)
    except Exception:
        pass
    mask_hns = df_final['padrao_tipo'].isin(['OCO', 'OCOI'])
    df_final['chave_idx'] = df_final['cabeca_idx'].where(
        mask_hns, df_final['p3_idx'])

    # Remove duplicates using the generic key
    df_final.drop_duplicates(subset=[
                             'ticker', 'timeframe', 'padrao_tipo', 'chave_idx'], inplace=True, keep='first')

    # Drop temporary key column
    df_final.drop(columns=['chave_idx'], inplace=True)

    cols_info = ['ticker', 'timeframe',
                 'strategy', 'padrao_tipo', 'score_total']
    cols_validacao = sorted(
        [col for col in df_final.columns if col.startswith('valid_')])
    cols_pontos = [
        col for col in df_final.columns if col.endswith(('_idx', '_preco'))]

    # Ensure all existing columns are included
    existing_cols = set(df_final.columns)
    ordem_final = [c for c in (
        cols_info + cols_validacao + cols_pontos) if c in existing_cols]

    df_final = df_final.reindex(columns=ordem_final)

    df_final.to_csv(final_csv_path, index=False,
                    date_format='%Y-%m-%d %H:%M:%S')
    print(f"\n{Fore.GREEN}✅ Final dataset with {len(df_final)} unique patterns saved to: {final_csv_path}")


if __name__ == "__main__":
    main()
