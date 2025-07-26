# --- gerador_de_dataset.py (v17 - VALIDAÇÃO AVANÇADA POR CONFLUÊNCIA) ---
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from typing import List, Dict, Any, Optional
import os
import time
from colorama import Fore, Style, init

# Inicializa o Colorama
init(autoreset=True)


class Config:
    """ Parâmetros de configuração para o detector de padrões. """
    TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'HBAR-USD']
    INTERVALS = ['1h', '4h', '1d']
    DATA_PERIOD = '5y'

    # Parâmetros únicos para o ZigZag (baseado exclusivamente em high/low)
    TIMEFRAME_PARAMS = {
        'default': {'depth': 2, 'deviation': 2.0},
        '1h':      {'depth': 3, 'deviation': 4.0},
        '4h':      {'depth': 3, 'deviation': 5.0},
        '1d':      {'depth': 5, 'deviation': 8.0}
    }

    # Parâmetros de validação geométrica
    SHOULDER_SYMMETRY_TOLERANCE = 0.10
    NECKLINE_FLATNESS_TOLERANCE = 0.10
    HEAD_SIGNIFICANCE_RATIO = 1.2
    HEAD_EXTREME_LOOKBACK_FACTOR = 5

    # Parâmetros de Download
    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 5

    # Caminho de saída final
    OUTPUT_DIR = 'data/datasets_hns_validados'
    FINAL_CSV_PATH = os.path.join(
        OUTPUT_DIR, 'dataset_hns_validado_confluencia.csv')

# --- FUNÇÕES DE VALIDAÇÃO DE INDICADORES ---


def check_rsi_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    """Verifica a divergência de RSI usando 'high' para OCO e 'low' para OCOI."""
    try:
        if tipo_padrao == 'OCO':
            rsi_series = ta.rsi(df['high'], length=14)
            if p1_idx not in rsi_series.index or p3_idx not in rsi_series.index:
                return False
            rsi_p1, rsi_p3 = rsi_series.loc[p1_idx], rsi_series.loc[p3_idx]
            return p3_price > p1_price and rsi_p3 < rsi_p1
        elif tipo_padrao == 'OCOI':
            rsi_series = ta.rsi(df['low'], length=14)
            if p1_idx not in rsi_series.index or p3_idx not in rsi_series.index:
                return False
            rsi_p1, rsi_p3 = rsi_series.loc[p1_idx], rsi_series.loc[p3_idx]
            return p3_price < p1_price and rsi_p3 > rsi_p1
    except Exception:
        return False
    return False


def check_macd_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    """Verifica a divergência do histograma do MACD."""
    try:
        macd = df.ta.macd(fast=12, slow=26, signal=9, append=False)
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


def check_volume_profile(df: pd.DataFrame, pivots: List[Dict[str, Any]], p1_idx, p3_idx, p5_idx) -> bool:
    """Verifica se o perfil de volume é ideal (ex: decrescente nos picos de um OCO)."""
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

# --- FUNÇÕES PRINCIPAIS ---


def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    original_period = period
    if 'm' in interval:
        period = '60d'
    elif 'h' in interval:
        period = '2y'
    if period != original_period:
        print(f"{Fore.YELLOW}Aviso: Período ajustado de '{original_period}' para '{period}' para o intervalo '{interval}'.{Style.RESET_ALL}")

    for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
        try:
            print(
                f"Buscando dados para {ticker}/{interval}... Tentativa {tentativa + 1}/{Config.MAX_DOWNLOAD_TENTATIVAS}")
            df = yf.download(tickers=ticker, period=period,
                             interval=interval, auto_adjust=True, progress=False)
            if not df.empty:
                print(f"{Fore.GREEN}Download bem-sucedido.{Style.RESET_ALL}")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [col.lower() for col in df.columns]
                return df
            else:
                raise ValueError("Download retornou um DataFrame vazio.")
        except Exception as e:
            print(
                f"{Fore.YELLOW}Falha na tentativa {tentativa + 1}: {e}{Style.RESET_ALL}")
            if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1:
                time.sleep(Config.RETRY_DELAY_SEGUNDOS)
            else:
                raise ConnectionError(
                    f"Download falhou para {ticker}/{interval} após {Config.MAX_DOWNLOAD_TENTATIVAS} tentativas.")
    raise ConnectionError(
        f"Falha inesperada no download para {ticker}/{interval}.")


def calcular_zigzag_oficial(df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
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
    return confirmed_pivots


def is_head_extreme(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_days: int) -> bool:
    """ Verifica se a cabeça do padrão é o ponto mais extremo num período de lookback. """
    lookback_period = int(avg_pivot_dist_days *
                          Config.HEAD_EXTREME_LOOKBACK_FACTOR)
    if lookback_period <= 0:
        return True
    start_date = head_pivot['idx'] - pd.Timedelta(days=lookback_period)
    end_date = head_pivot['idx'] + pd.Timedelta(days=lookback_period)
    context_df = df.loc[start_date:end_date]
    if context_df.empty:
        return True
    if head_pivot['tipo'] == 'PICO':
        return head_pivot['preco'] >= context_df['high'].max()
    else:
        return head_pivot['preco'] <= context_df['low'].min()


def _validate_hns_pattern(p0: Dict, p1: Dict, p2: Dict, p3: Dict, p4: Dict, p5: Dict, tipo_padrao: str, df_historico: pd.DataFrame, avg_pivot_dist: int) -> Optional[Dict[str, Any]]:
    ombro_esq, neckline1, cabeca, neckline2, ombro_dir = p1, p2, p3, p4, p5
    if tipo_padrao == 'OCO':
        if not (cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']):
            return None
        if not (ombro_esq['preco'] > neckline1['preco'] and ombro_dir['preco'] > neckline2['preco']):
            return None
        if not (ombro_dir['preco'] < ombro_esq['preco']):
            return None
        if not (p0['preco'] < neckline1['preco'] and p0['preco'] < neckline2['preco']):
            return None
    elif tipo_padrao == 'OCOI':
        if not (cabeca['preco'] < ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco']):
            return None
        if not (ombro_esq['preco'] < neckline1['preco'] and ombro_dir['preco'] < neckline2['preco']):
            return None
        if not (ombro_dir['preco'] > ombro_esq['preco']):
            return None
        if not (p0['preco'] > neckline1['preco'] and p0['preco'] > neckline2['preco']):
            return None
    altura_cabeca = abs(
        cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_esq = abs(
        ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_dir = abs(
        ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    if altura_ombro_esq == 0 or altura_cabeca == 0:
        return None
    if altura_cabeca / altura_ombro_esq < Config.HEAD_SIGNIFICANCE_RATIO:
        return None
    if abs(altura_ombro_esq - altura_ombro_dir) > altura_cabeca * Config.SHOULDER_SYMMETRY_TOLERANCE:
        return None
    if abs(neckline1['preco'] - neckline2['preco']) > altura_ombro_esq * Config.NECKLINE_FLATNESS_TOLERANCE:
        return None
    if not is_head_extreme(df_historico, cabeca, avg_pivot_dist):
        return None

    # Retorna todos os pontos para a validação de volume
    return {'padrao_tipo': tipo_padrao, 'p0_idx': p0['idx'], 'ombro1_idx': p1['idx'], 'ombro1_preco': p1['preco'], 'neckline1_idx': p2['idx'], 'neckline1_preco': p2['preco'], 'cabeca_idx': p3['idx'], 'cabeca_preco': p3['preco'], 'neckline2_idx': p4['idx'], 'neckline2_preco': p4['preco'], 'ombro2_idx': p5['idx'], 'ombro2_preco': p5['preco']}


def identificar_padroes_hns(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    padroes_encontrados = []
    n = len(pivots)
    if n < 7:
        return []
    avg_pivot_dist_days = np.mean(
        [(pivots[i]['idx'] - pivots[i-1]['idx']).days for i in range(1, n)]) if n > 1 else 0
    for i in range(n - 6):
        janela = pivots[i:i+7]
        p0, p1, p2, p3, p4, p5 = janela[0], janela[1], janela[2], janela[3], janela[4], janela[5]
        tipo_padrao = None
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'OCO'
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'OCOI'
        if tipo_padrao:
            dados_padrao = _validate_hns_pattern(
                p0, p1, p2, p3, p4, p5, tipo_padrao, df_historico, avg_pivot_dist_days)
            if dados_padrao:
                rsi_ok = check_rsi_divergence(
                    df_historico, dados_padrao['ombro1_idx'], dados_padrao['cabeca_idx'], dados_padrao['ombro1_preco'], dados_padrao['cabeca_preco'], tipo_padrao)
                macd_ok = check_macd_divergence(
                    df_historico, dados_padrao['ombro1_idx'], dados_padrao['cabeca_idx'], dados_padrao['ombro1_preco'], dados_padrao['cabeca_preco'], tipo_padrao)
                volume_ok = check_volume_profile(
                    df_historico, pivots, dados_padrao['ombro1_idx'], dados_padrao['cabeca_idx'], dados_padrao['ombro2_idx'])
                if rsi_ok and macd_ok and volume_ok:
                    dados_padrao['rsi_confirmou'], dados_padrao['macd_confirmou'], dados_padrao['volume_confirmou'] = rsi_ok, macd_ok, volume_ok
                    padroes_encontrados.append(dados_padrao)
    return padroes_encontrados


def main():
    print(f"{Style.BRIGHT}--- INICIANDO MOTOR DE GERAÇÃO (v17 - Validação Avançada) ---")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    todos_os_padroes_finais = []

    for ticker in Config.TICKERS:
        for interval in Config.INTERVALS:
            print(f"\n--- Processando: {ticker} | Intervalo: {interval} ---")
            try:
                params = Config.TIMEFRAME_PARAMS.get(
                    interval, Config.TIMEFRAME_PARAMS['default'])
                df_historico = buscar_dados(
                    ticker, Config.DATA_PERIOD, interval)
                pivots_detectados = calcular_zigzag_oficial(
                    df_historico, params['depth'], params['deviation'])
                if len(pivots_detectados) < 7:
                    print("ℹ️ Número insuficiente de pivôs para formar um padrão.")
                    continue
                padroes_encontrados = identificar_padroes_hns(
                    pivots_detectados, df_historico)
                if padroes_encontrados:
                    print(
                        f"{Fore.GREEN}✅ Encontrados {len(padroes_encontrados)} padrões validados por confluência.")
                    for padrao in padroes_encontrados:
                        padrao['ticker'], padrao['timeframe'] = ticker, interval
                        todos_os_padroes_finais.append(padrao)
                else:
                    print("ℹ️ Nenhum padrão passou em todos os critérios de validação.")
            except Exception as e:
                print(f"{Fore.RED}❌ Erro ao processar {ticker}/{interval}: {e}")

    print(f"\n{Style.BRIGHT}--- Processo finalizado. Salvando dataset validado... ---{Style.RESET_ALL}")

    if not todos_os_padroes_finais:
        print(f"{Fore.YELLOW}Nenhum padrão foi encontrado em todas as execuções.")
        return

    df_final = pd.DataFrame(todos_os_padroes_finais)
    df_final.to_csv(Config.FINAL_CSV_PATH, index=False,
                    date_format='%Y-%m-%d %H:%M:%S')
    print(f"\n{Fore.GREEN}✅ Dataset final com {len(df_final)} padrões únicos e validados salvo em: {Config.FINAL_CSV_PATH}")


if __name__ == "__main__":
    main()
