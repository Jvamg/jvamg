# --- MOTOR DE GERAÇÃO DE DATASET OCO - v9 (LÓGICA FIEL AO PINE SCRIPT) ---
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional
import os
from colorama import Fore, Style, init

# Inicializa o Colorama
init(autoreset=True)

# --- PASSO 1: CONFIGURAÇÃO CENTRALIZADA E ADAPTÁVEL ---


class Config:
    """ Parâmetros de configuração para o detector de padrões. """
    TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'HBAR-USD']
    INTERVALS = ['15m', '1h', '4h', '1d']
    DATA_PERIOD = '5y'

    TIMEFRAME_PARAMS = {
        'default': {'deviation': 4.0, 'min_distance': 5},  # Para 15m, 30m etc.
        '1h':      {'deviation': 7.0, 'min_distance': 5},
        '4h':      {'deviation': 10.0, 'min_distance': 5},
        '1d':      {'deviation': 15.0, 'min_distance': 7}
    }

    # --- PARÂMETROS DE VALIDAÇÃO (AJUSTADOS PARA FICAREM IGUAIS AO PINE SCRIPT) ---
    # MUDANÇA: De 0.30 para 0.10 (muito mais rigoroso)
    SHOULDER_SYMMETRY_TOLERANCE = 0.10
    # MUDANÇA: De 0.25 para 0.10 (muito mais rigoroso)
    NECKLINE_FLATNESS_TOLERANCE = 0.10
    HEAD_SIGNIFICANCE_RATIO = 1.2
    HEAD_EXTREME_LOOKBACK_FACTOR = 5

    OUTPUT_CSV_PATH = 'data/datasets_hns/hns_final_adaptive_dataset.csv'

# --- FUNÇÕES DE DETECÇÃO ---


def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    original_period = period
    if 'm' in interval:
        period = '60d'
    elif 'h' in interval:
        period = '2y'
    if period != original_period:
        print(f"{Fore.YELLOW}Aviso: Período de busca ajustado de '{original_period}' para '{period}' para o intervalo '{interval}'.{Style.RESET_ALL}")

    df = yf.download(tickers=ticker, period=period,
                     interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ConnectionError(
            f"Não foi possível buscar dados para {ticker}/{interval}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [col.lower() for col in df.columns]
    return df


def calcular_zigzag(df: pd.DataFrame, deviation_percent: float, min_distance_bars: int) -> List[Dict[str, Any]]:
    if len(df) < 2:
        return []
    pivots = [{'idx': df.index[0], 'preco': df['low'].iloc[0], 'tipo': 'VALE'}]
    trend, peak, valley = 'UP', {'preco': -np.inf}, {'preco': np.inf}
    for i in range(1, len(df)):
        row, idx = df.iloc[i], df.index[i]
        if trend == 'UP':
            if row['high'] > peak['preco']:
                peak = {'preco': row['high'], 'idx': idx}
            if peak['preco'] > 0 and (peak['preco'] - row['low']) / peak['preco'] > deviation_percent / 100:
                pivots.append(
                    {'idx': peak['idx'], 'preco': peak['preco'], 'tipo': 'PICO'})
                trend, valley = 'DOWN', {'preco': row['low'], 'idx': idx}
        elif trend == 'DOWN':
            if row['low'] < valley['preco']:
                valley = {'preco': row['low'], 'idx': idx}
            if valley['preco'] > 0 and (row['high'] - valley['preco']) / valley['preco'] > deviation_percent / 100:
                pivots.append(
                    {'idx': valley['idx'], 'preco': valley['preco'], 'tipo': 'VALE'})
                trend, peak = 'UP', {'preco': row['high'], 'idx': idx}
    if not pivots:
        return []
    pivots_filtrados = [pivots[0]]
    for i in range(1, len(pivots)):
        try:
            dist = abs(df.index.get_loc(
                pivots[i]['idx']) - df.index.get_loc(pivots_filtrados[-1]['idx']))
            if pivots[i]['tipo'] != pivots_filtrados[-1]['tipo'] and dist >= min_distance_bars:
                pivots_filtrados.append(pivots[i])
        except KeyError:
            continue
    return pivots_filtrados


def is_head_extreme(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_days: int) -> bool:
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


def _validate_hns_pattern(
    p0: Dict, p1: Dict, p2: Dict, p3: Dict, p4: Dict, p5: Dict,
    tipo_padrao: str, df_historico: pd.DataFrame, avg_pivot_dist: int
) -> Optional[Dict[str, Any]]:
    # --- LÓGICA DE VALIDAÇÃO COMPLETA E RIGOROSA INSERIDA AQUI ---
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

    return {'padrao_tipo': tipo_padrao, 'ombro1_idx': ombro_esq['idx'], 'ombro1_preco': ombro_esq['preco'], 'neckline1_idx': neckline1['idx'], 'neckline1_preco': neckline1['preco'], 'cabeca_idx': cabeca['idx'], 'cabeca_preco': cabeca['preco'], 'neckline2_idx': neckline2['idx'], 'neckline2_preco': neckline2['preco'], 'ombro2_idx': ombro_dir['idx'], 'ombro2_preco': ombro_dir['preco']}


def identificar_padroes_hns(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    padroes_encontrados = []
    n = len(pivots)
    if n < 7:
        return []
    datas_pivots = [p['idx'] for p in pivots]
    avg_pivot_dist_days = np.mean(
        [(datas_pivots[i] - datas_pivots[i-1]).days for i in range(1, n)]) if n > 1 else 0
    for i in range(n - 6):
        janela = pivots[i:i+7]
        p0, p1, p2, p3, p4, p5, p6 = janela
        tipo_padrao = None
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'OCO'
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'OCOI'
        if tipo_padrao:
            dados_padrao = _validate_hns_pattern(
                p0, p1, p2, p3, p4, p5, tipo_padrao, df_historico, avg_pivot_dist_days)
            if dados_padrao:
                padroes_encontrados.append(dados_padrao)
    return padroes_encontrados


def main():
    print(f"{Style.BRIGHT}--- INICIANDO MOTOR DE GERAÇÃO DE DATASET OCO (v9 - Lógica Fiel) ---")
    todos_os_padroes = []
    for ticker in Config.TICKERS:
        for interval in Config.INTERVALS:
            print(
                f"\n{Style.BRIGHT}--- Processando: {ticker} | Intervalo: {interval} ---{Style.RESET_ALL}")
            try:
                params = Config.TIMEFRAME_PARAMS.get(
                    interval, Config.TIMEFRAME_PARAMS['default'])
                df_historico = buscar_dados(
                    ticker, Config.DATA_PERIOD, interval)
                pivots_detectados = calcular_zigzag(
                    df_historico, params['deviation'], params['min_distance'])

                if len(pivots_detectados) < 7:
                    print("ℹ️ Número insuficiente de pivôs para formar um padrão.")
                    continue

                # A identificação agora usa a validação rigorosa
                padroes_encontrados = identificar_padroes_hns(
                    pivots_detectados, df_historico)

                if padroes_encontrados:
                    print(
                        f"{Fore.GREEN}✅ Encontrados {len(padroes_encontrados)} padrões de alta qualidade.")
                    for padrao in padroes_encontrados:
                        padrao['ticker'], padrao['timeframe'] = ticker, interval
                        todos_os_padroes.append(padrao)
                else:
                    print("ℹ️ Nenhum padrão com as regras rigorosas foi encontrado.")
            except Exception as e:
                print(f"{Fore.RED}❌ Erro ao processar {ticker}/{interval}: {e}")

    if not todos_os_padroes:
        print(f"\n{Fore.YELLOW}Processo concluído. Nenhum padrão foi encontrado.")
        return

    print(f"\n{Style.BRIGHT}{Fore.GREEN}Análise finalizada! Total de {len(todos_os_padroes)} padrões de alta qualidade encontrados.{Style.RESET_ALL}")
    df_final = pd.DataFrame(todos_os_padroes)
    colunas_ordenadas = ['ticker', 'timeframe', 'padrao_tipo', 'ombro1_idx', 'ombro1_preco', 'neckline1_idx',
                         'neckline1_preco', 'cabeca_idx', 'cabeca_preco', 'neckline2_idx', 'neckline2_preco', 'ombro2_idx', 'ombro2_preco']
    df_final = df_final.reindex(columns=colunas_ordenadas)
    try:
        os.makedirs(os.path.dirname(Config.OUTPUT_CSV_PATH), exist_ok=True)
        df_final.to_csv(Config.OUTPUT_CSV_PATH, index=False,
                        date_format='%Y-%m-%d %H:%M:%S')
        print(
            f"Dataset salvo com sucesso em: {Style.BRIGHT}{Config.OUTPUT_CSV_PATH}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Erro ao salvar o arquivo CSV: {e}")


if __name__ == "__main__":
    main()
