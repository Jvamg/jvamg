import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Import the module under test
import src.patterns.OCOs.necklineconfirmada as nc


@pytest.fixture
def df_index():
    return pd.date_range(start="2023-01-01", periods=100, freq="D")


def make_df(index: pd.DatetimeIndex,
            close: np.ndarray,
            high: np.ndarray | None = None,
            low: np.ndarray | None = None,
            open_: np.ndarray | None = None,
            volume: np.ndarray | None = None) -> pd.DataFrame:
    n = len(index)
    df = pd.DataFrame(index=index)
    df['close'] = close
    df['open'] = open_ if open_ is not None else close.copy()
    # default high/low around close if not provided
    df['high'] = high if high is not None else (close + 0.5)
    df['low'] = low if low is not None else (close - 0.5)
    df['volume'] = volume if volume is not None else np.full(n, 100.0)
    return df


# assess_rsi_divergence_strength tests

def test_rsi_divergence_neutral_zone_returns_false(df_index, monkeypatch):
    df = make_df(df_index, close=np.linspace(100, 110, len(df_index)))

    # Provide a fake RSI series: neutral start at p1
    p1_idx = df.index[50]
    p3_idx = df.index[60]
    rsi_series = pd.Series(index=df.index, data=50.0)
    rsi_series.loc[p1_idx] = 60.0  # not >= 70
    rsi_series.loc[p3_idx] = 55.0

    monkeypatch.setattr(nc.ta, 'rsi', lambda s, length=14: rsi_series)

    div, strong = nc.assess_rsi_divergence_strength(
        df, p1_idx, p3_idx, p1_price=100, p3_price=110,
        direction='bearish', source_series=df['high']
    )
    assert div == False and strong == False


def test_rsi_divergence_weak_and_strong(df_index, monkeypatch):
    df = make_df(df_index, close=np.linspace(100, 110, len(df_index)))
    p1_idx = df.index[40]
    p3_idx = df.index[70]

    # Weak divergence: start >= 70, delta < min_delta
    rsi_series1 = pd.Series(index=df.index, data=50.0)
    rsi_series1.loc[p1_idx] = 72.0
    rsi_series1.loc[p3_idx] = 69.0  # delta 3 < 5

    monkeypatch.setattr(nc.ta, 'rsi', lambda s, length=14: rsi_series1)
    div, strong = nc.assess_rsi_divergence_strength(
        df, p1_idx, p3_idx, p1_price=100, p3_price=110,
        direction='bearish', source_series=df['high']
    )
    assert div == True and strong == False

    # Strong divergence: start >= 80 or delta >= min_delta
    rsi_series2 = pd.Series(index=df.index, data=50.0)
    rsi_series2.loc[p1_idx] = 85.0
    rsi_series2.loc[p3_idx] = 75.0  # delta 10

    monkeypatch.setattr(nc.ta, 'rsi', lambda s, length=14: rsi_series2)
    div2, strong2 = nc.assess_rsi_divergence_strength(
        df, p1_idx, p3_idx, p1_price=100, p3_price=110,
        direction='bearish', source_series=df['high']
    )
    assert div2 == True and strong2 == True


# detect_macd_signal_cross tests

def test_detect_macd_signal_cross_valid_and_invalid(df_index, monkeypatch):
    df = make_df(df_index, close=np.linspace(100, 120, len(df_index)))

    macd_df = pd.DataFrame(index=df.index)
    macd_df['MACD_12_26_9'] = 0.10
    macd_df['MACDs_12_26_9'] = 0.05
    macd_df.iloc[-1, macd_df.columns.get_loc('MACD_12_26_9')] = -0.05
    macd_df.iloc[-1, macd_df.columns.get_loc('MACDs_12_26_9')] = 0.00

    # Patch df.ta.macd
    monkeypatch.setattr(df.ta, 'macd', lambda *args, **kwargs: macd_df)

    idx_ref = df.index[-1]
    assert nc.detect_macd_signal_cross(
        df, idx_ref, direction='bearish') == True
    assert nc.detect_macd_signal_cross(
        df, idx_ref, direction='bullish') == False

    macd_df2 = macd_df.copy()
    macd_df2.iloc[-2, macd_df2.columns.get_loc('MACD_12_26_9')] = -0.05
    macd_df2.iloc[-2, macd_df2.columns.get_loc('MACDs_12_26_9')] = 0.00
    macd_df2.iloc[-1, macd_df2.columns.get_loc('MACD_12_26_9')] = 0.05
    macd_df2.iloc[-1, macd_df2.columns.get_loc('MACDs_12_26_9')] = 0.00

    monkeypatch.setattr(df.ta, 'macd', lambda *args, **kwargs: macd_df2)

    assert nc.detect_macd_signal_cross(
        df, idx_ref, direction='bullish') == True
    assert nc.detect_macd_signal_cross(
        df, idx_ref, direction='bearish') == False


# check_stochastic_confirmation tests

def test_stochastic_confirmation_requires_ob_os_and_cross(df_index, monkeypatch):
    df = make_df(df_index, close=np.linspace(100, 110, len(df_index)))
    p1_idx = df.index[-10]
    p3_idx = df.index[-1]

    k_col = f"STOCHk_{getattr(nc.Config, 'STOCH_K', 14)}_{getattr(nc.Config, 'STOCH_D', 3)}_{getattr(nc.Config, 'STOCH_SMOOTH_K', 3)}"
    d_col = f"STOCHd_{getattr(nc.Config, 'STOCH_K', 14)}_{getattr(nc.Config, 'STOCH_D', 3)}_{getattr(nc.Config, 'STOCH_SMOOTH_K', 3)}"

    # Case 1: start outside OB/OS -> no confirmations
    stoch_df1 = pd.DataFrame(index=df.index, data=0.0, columns=[k_col, d_col])
    stoch_df1.loc[p1_idx, k_col] = 50.0
    stoch_df1.loc[p3_idx, k_col] = 55.0
    # cross diff changes not relevant because gating fails
    stoch_df1.loc[df.index[-2], k_col] = 60.0
    stoch_df1.loc[df.index[-2], d_col] = 55.0
    stoch_df1.loc[df.index[-1], k_col] = 55.0
    stoch_df1.loc[df.index[-1], d_col] = 60.0

    monkeypatch.setattr(nc.ta, 'stoch', lambda **kwargs: stoch_df1)
    res = nc.check_stochastic_confirmation(
        df, p1_idx, p3_idx, 100, 110, direction='bearish')
    assert res['valid_estocastico_divergencia'] == False
    assert res['valid_estocastico_cross'] == False

    # Case 2: OB start, divergence + bearish cross valid
    stoch_df2 = pd.DataFrame(index=df.index, data=0.0, columns=[k_col, d_col])
    stoch_df2.loc[p1_idx, k_col] = 85.0  # OB
    stoch_df2.loc[p3_idx, k_col] = 70.0  # lower K with p3_price > p1_price
    # Cross: prev diff >= 0, curr diff < 0
    stoch_df2.loc[df.index[-2], k_col] = 90.0
    stoch_df2.loc[df.index[-2], d_col] = 85.0
    stoch_df2.loc[df.index[-1], k_col] = 60.0
    stoch_df2.loc[df.index[-1], d_col] = 65.0

    monkeypatch.setattr(nc.ta, 'stoch', lambda **kwargs: stoch_df2)
    res2 = nc.check_stochastic_confirmation(
        df, p1_idx, p3_idx, 100, 110, direction='bearish')
    assert res2['valid_estocastico_divergencia'] == True
    assert res2['valid_estocastico_cross'] == True

    # Case 3: OB start, divergence true but no cross
    stoch_df3 = stoch_df2.copy()
    stoch_df3.loc[df.index[-2], k_col] = 60.0
    stoch_df3.loc[df.index[-2], d_col] = 61.0
    stoch_df3.loc[df.index[-1], k_col] = 59.0
    stoch_df3.loc[df.index[-1], d_col] = 60.0

    monkeypatch.setattr(nc.ta, 'stoch', lambda **kwargs: stoch_df3)
    res3 = nc.check_stochastic_confirmation(
        df, p1_idx, p3_idx, 100, 110, direction='bearish')
    assert res3['valid_estocastico_divergencia'] == True
    assert res3['valid_estocastico_cross'] == False


# find_breakout_index + check_breakout_volume tests

def test_breakout_and_volume(df_index):
    # Prices stable above neckline then break below
    close = np.full(len(df_index), 100.0)
    close[10:] = 99.0
    close[15] = 97.0  # breakout bar (bearish below neckline 98)
    volume = np.full(len(df_index), 100.0)
    volume[15] = 200.0  # 2x baseline
    df = make_df(df_index, close=close, volume=volume)

    start_idx = df.index[5]
    neckline = 98.0

    br_idx = nc.find_breakout_index(
        df, neckline, start_idx, direction='bearish', max_bars=30)
    assert br_idx == df.index[15]
    assert nc.check_breakout_volume(df, br_idx) == True

    # Insufficient volume
    df2 = df.copy()
    df2.loc[df.index[15], 'volume'] = 150.0  # < 1.8x of 100
    br_idx2 = nc.find_breakout_index(
        df2, neckline, start_idx, direction='bearish', max_bars=30)
    assert br_idx2 == df.index[15]
    assert nc.check_breakout_volume(df2, br_idx2) == False

    # No breakout
    close3 = np.full(len(df_index), 100.0)
    df3 = make_df(df_index, close=close3, volume=np.full(len(df_index), 100.0))
    assert nc.find_breakout_index(
        df3, neckline, start_idx, direction='bearish', max_bars=30) is None


# Integration with validators

def _make_simple_hns_df_and_pivots(df_index):
    # Construct OCO pattern pivots: ['VALE','PICO','VALE','PICO','VALE','PICO','VALE']
    # Prices designed to satisfy symmetry/flat neckline and prominence
    close = np.linspace(100.0, 105.0, len(df_index))
    df = make_df(df_index, close=close)

    p0 = {'idx': df.index[10], 'tipo': 'VALE', 'preco': 95.0}
    p1 = {'idx': df.index[20], 'tipo': 'PICO', 'preco': 105.0}
    p2 = {'idx': df.index[30], 'tipo': 'VALE', 'preco': 100.0}
    p3 = {'idx': df.index[40], 'tipo': 'PICO', 'preco': 110.0}
    p4 = {'idx': df.index[50], 'tipo': 'VALE', 'preco': 101.0}
    p5 = {'idx': df.index[60], 'tipo': 'PICO', 'preco': 104.0}
    # retest exactly at neckline mean ensures tolerance passes
    neckline_price = (p2['preco'] + p4['preco']) / 2.0
    p6 = {'idx': df.index[70], 'tipo': 'VALE', 'preco': neckline_price}

    pivots = [p0, p1, p2, p3, p4, p5, p6]
    avg_bars = 10

    return df, pivots, (p0, p1, p2, p3, p4, p5, p6), avg_bars


def test_validate_and_score_hns_integration(df_index, monkeypatch):
    df, pivots, (p0, p1, p2, p3, p4, p5,
                 p6), avg_bars = _make_simple_hns_df_and_pivots(df_index)

    # Patch gates and optional indicators to deterministic values
    monkeypatch.setattr(nc, 'is_head_extreme', lambda *args, **kwargs: True)

    # RSI strong
    monkeypatch.setattr(nc, 'assess_rsi_divergence_strength',
                        lambda *args, **kwargs: (True, True))
    # MACD histogram divergence
    monkeypatch.setattr(nc, 'check_macd_divergence',
                        lambda *args, **kwargs: True)
    # MACD signal cross
    monkeypatch.setattr(nc, 'detect_macd_signal_cross',
                        lambda *args, **kwargs: True)
    # Stochastic confirmations
    monkeypatch.setattr(nc, 'check_stochastic_confirmation', lambda *args, **kwargs: {
        'valid_estocastico_divergencia': True,
        'valid_estocastico_cross': True,
    })
    # Breakout index + volume
    monkeypatch.setattr(nc, 'find_breakout_index',
                        lambda *args, **kwargs: df.index[75])
    monkeypatch.setattr(nc, 'check_breakout_volume',
                        lambda *args, **kwargs: True)
    # Volume profile (optional) set False to keep score deterministic
    monkeypatch.setattr(nc, 'check_volume_profile',
                        lambda *args, **kwargs: False)

    data = nc.validate_and_score_hns_pattern(
        *pivots, 'OCO', df, pivots, avg_bars)
    assert isinstance(data, dict)

    # Expected flags
    for flag in [
        'valid_divergencia_rsi', 'valid_divergencia_rsi_strong',
        'valid_divergencia_macd', 'valid_macd_signal_cross',
        'valid_estocastico_divergencia', 'valid_estocastico_cross',
        'valid_volume_breakout_neckline', 'valid_proeminencia_cabeca',
        'valid_neckline_plana', 'valid_neckline_retest_p6',
    ]:
        assert data.get(flag) == True

    # Score equals sum of mandatory + these optionals + base trend/symmetry/ombro_fraco
    w = nc.Config.SCORE_WEIGHTS_HNS
    expected = (
        w['valid_extremo_cabeca'] + w['valid_contexto_cabeca'] +
        w['valid_simetria_ombros'] + w['valid_neckline_plana'] +
        w['valid_base_tendencia'] + w['valid_neckline_retest_p6'] +
        w['valid_divergencia_rsi'] + w['valid_divergencia_rsi_strong'] +
        w['valid_divergencia_macd'] + w['valid_macd_signal_cross'] +
        w['valid_estocastico_divergencia'] + w['valid_estocastico_cross'] +
        w['valid_ombro_direito_fraco'] + w['valid_proeminencia_cabeca'] +
        w['valid_volume_breakout_neckline']
    )
    assert data['score_total'] == expected


def _make_simple_dt_df_and_pivots(df_index):
    # DT pivots: ['VALE','PICO','VALE','PICO','VALE']
    close = np.linspace(100.0, 105.0, len(df_index))
    df = make_df(df_index, close=close)

    p0 = {'idx': df.index[10], 'tipo': 'VALE', 'preco': 95.0}
    p1 = {'idx': df.index[20], 'tipo': 'PICO', 'preco': 105.0}
    p2 = {'idx': df.index[30], 'tipo': 'VALE', 'preco': 98.0}
    p3 = {'idx': df.index[40], 'tipo': 'PICO',
          'preco': 104.5}  # within symmetry tolerance
    p4 = {'idx': df.index[50], 'tipo': 'VALE',
          'preco': 98.0}   # retest exactly
    pivots = [p0, p1, p2, p3, p4]
    avg_bars = 10
    return df, pivots, (p0, p1, p2, p3, p4), avg_bars


def test_validate_and_score_dtb_integration(df_index, monkeypatch):
    df, pivots, (p0, p1, p2, p3,
                 p4), avg_bars = _make_simple_dt_df_and_pivots(df_index)

    monkeypatch.setattr(nc, 'check_volume_profile_dtb',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(nc, 'check_obv_divergence_dtb',
                        lambda *args, **kwargs: False)

    monkeypatch.setattr(nc, 'assess_rsi_divergence_strength',
                        lambda *args, **kwargs: (True, True))
    # DTB now uses MACD divergence via histogram for HNS only and optional confirmations via signal cross.
    # Mock MACD divergence path by replacing check_macd_divergence (used in HNS) not needed here;
    # keep signal cross true to assert 'valid_macd_signal_cross' flag.
    monkeypatch.setattr(nc, 'detect_macd_signal_cross',
                        lambda *args, **kwargs: True)  # <-- Importante!
    monkeypatch.setattr(nc, 'check_stochastic_confirmation', lambda *args, **kwargs: {
        'valid_estocastico_divergencia': True,
        'valid_estocastico_cross': True,
    })
    monkeypatch.setattr(nc, 'find_breakout_index',
                        lambda *args, **kwargs: df.index[55])
    monkeypatch.setattr(nc, 'check_breakout_volume',
                        lambda *args, **kwargs: True)

    data = nc.validate_and_score_double_pattern(*pivots, 'DT', df, avg_bars)
    assert isinstance(data, dict)

    for flag in [
        'valid_estrutura_picos_vales', 'valid_simetria_extremos',
        'valid_profundidade_vale_pico', 'valid_contexto_extremos',
        'valid_contexto_tendencia', 'valid_neckline_retest_p4',
        'valid_divergencia_rsi', 'valid_divergencia_rsi_strong',
        'valid_divergencia_macd', 'valid_macd_signal_cross',  # Aqui deve retornar True
        'valid_estocastico_divergencia', 'valid_estocastico_cross',
        'valid_volume_breakout_neckline',
    ]:
        assert data.get(flag) == True
