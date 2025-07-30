# --- gerador_de_dataset.py (v19 - REGRAS OBRIGATÓRIAS + SCORING) ---
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
    TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'HBAR-USD']
    INTERVALS = ['1h', '4h', '1d']
    DATA_PERIOD = '5y'
    
    TIMEFRAME_PARAMS = {
        'default': {'depth': 2, 'deviation': 2.0},
        '1h':      {'depth': 3, 'deviation': 4.0},
        '4h':      {'depth': 3, 'deviation': 5.0},
        '1d':      {'depth': 5, 'deviation': 8.0}
    }

    # --- SISTEMA DE REGRAS E PONTUAÇÃO ---
    # As 5 regras a seguir são OBRIGATÓRIAS. Um padrão só é considerado se TODAS forem verdadeiras.
    # 'valid_extremo_cabeca', 'valid_contexto_cabeca', 'valid_simetria_ombros', 
    # 'valid_neckline_plana', 'valid_base_tendencia'
    #
    # Se um padrão passar, ele recebe uma pontuação com base nos pesos abaixo.
    SCORE_WEIGHTS = {
        # --- Regras Obrigatórias (contribuem para o score) ---
        'valid_extremo_cabeca': 20,
        'valid_contexto_cabeca': 15,
        'valid_simetria_ombros': 10,
        'valid_neckline_plana': 5,
        'valid_base_tendencia': 5,
        # --- Regras Opcionais (adicionam pontos) ---
        'valid_divergencia_rsi': 15,
        'valid_divergencia_macd': 10,
        'valid_proeminencia_cabeca': 10,
        'valid_ombro_direito_fraco': 5,
        'valid_perfil_volume': 5
    }
    MINIMUM_SCORE_TO_SAVE = 70

    # Parâmetros de validação
    HEAD_SIGNIFICANCE_RATIO = 1.2
    SHOULDER_SYMMETRY_TOLERANCE = 0.10
    NECKLINE_FLATNESS_TOLERANCE = 0.25
    HEAD_EXTREME_LOOKBACK_FACTOR = 5
    
    MAX_DOWNLOAD_TENTATIVAS, RETRY_DELAY_SEGUNDOS = 3, 5
    OUTPUT_DIR = 'data/datasets_hns_scored_mandatory'
    FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, 'dataset_hns_scored_mandatory_final.csv')

# --- FUNÇÕES AUXILIARES ---
def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # ... (código idêntico)
    original_period = period
    if 'm' in interval: period = '60d'
    elif 'h' in interval: period = '2y'
    if period != original_period: print(f"{Fore.YELLOW}Aviso: Período ajustado de '{original_period}' para '{period}' para o intervalo '{interval}'.{Style.RESET_ALL}")
    for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
        try:
            df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df.columns = [col.lower() for col in df.columns]
                return df
            else: raise ValueError("Download retornou um DataFrame vazio.")
        except Exception as e:
            if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1: time.sleep(Config.RETRY_DELAY_SEGUNDOS)
            else: raise ConnectionError(f"Download falhou para {ticker}/{interval} após {Config.MAX_DOWNLOAD_TENTATIVAS} tentativas.")
    raise ConnectionError(f"Falha inesperada no download para {ticker}/{interval}.")

def calcular_zigzag_oficial(df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
    # ... (código idêntico, usando apenas high/low)
    peak_series, valley_series = df['high'], df['low']
    window_size = 2 * depth + 1
    rolling_max, rolling_min = peak_series.rolling(window=window_size, center=True, min_periods=1).max(), valley_series.rolling(window=window_size, center=True, min_periods=1).min()
    candidate_peaks_df, candidate_valleys_df = df[peak_series == rolling_max], df[valley_series == rolling_min]
    candidates = []
    for idx, row in candidate_peaks_df.iterrows(): candidates.append({'idx': idx, 'preco': row[peak_series.name], 'tipo': 'PICO'})
    for idx, row in candidate_valleys_df.iterrows(): candidates.append({'idx': idx, 'preco': row[valley_series.name], 'tipo': 'VALE'})
    candidates = sorted(list({p['idx']: p for p in candidates}.values()), key=lambda x: x['idx'])
    if len(candidates) < 2: return []
    confirmed_pivots = [candidates[0]]
    last_pivot = candidates[0]
    for i in range(1, len(candidates)):
        candidate = candidates[i]
        if candidate['tipo'] == last_pivot['tipo']:
            if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']): confirmed_pivots[-1], last_pivot = candidate, candidate
            continue
        if last_pivot['preco'] == 0: continue
        price_dev = abs(candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
        if price_dev >= deviation_percent: confirmed_pivots.append(candidate); last_pivot = candidate
    return confirmed_pivots

def is_head_extreme(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_days: int) -> bool:
    # ... (código idêntico)
    lookback_period = int(avg_pivot_dist_days * Config.HEAD_EXTREME_LOOKBACK_FACTOR)
    if lookback_period <= 0: return True
    start_date, end_date = head_pivot['idx'] - pd.Timedelta(days=lookback_period), head_pivot['idx'] + pd.Timedelta(days=lookback_period)
    context_df = df.loc[start_date:end_date]
    if context_df.empty: return True
    if head_pivot['tipo'] == 'PICO': return head_pivot['preco'] >= context_df['high'].max()
    else: return head_pivot['preco'] <= context_df['low'].min()

def check_rsi_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    # ... (código idêntico)
    try:
        if tipo_padrao == 'OCO':
            rsi_series = ta.rsi(df['high'], length=14)
            if p1_idx not in rsi_series.index or p3_idx not in rsi_series.index: return False
            rsi_p1, rsi_p3 = rsi_series.loc[p1_idx], rsi_series.loc[p3_idx]
            return p3_price > p1_price and rsi_p3 < rsi_p1
        elif tipo_padrao == 'OCOI':
            rsi_series = ta.rsi(df['low'], length=14)
            if p1_idx not in rsi_series.index or p3_idx not in rsi_series.index: return False
            rsi_p1, rsi_p3 = rsi_series.loc[p1_idx], rsi_series.loc[p3_idx]
            return p3_price < p1_price and rsi_p3 > rsi_p1
    except Exception: return False
    return False

def check_macd_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    # ... (código idêntico)
    try:
        macd = df.ta.macd(fast=12, slow=26, signal=9, append=False)
        hist_col = 'MACDh_12_26_9'
        if p1_idx not in macd.index or p3_idx not in macd.index: return False
        hist_p1, hist_p3 = macd[hist_col].loc[p1_idx], macd[hist_col].loc[p3_idx]
        if tipo_padrao == 'OCO': return p3_price > p1_price and hist_p3 < hist_p1
        elif tipo_padrao == 'OCOI': return p3_price < p1_price and hist_p3 > hist_p1
    except Exception: return False
    return False

def check_volume_profile(df: pd.DataFrame, pivots: List[Dict[str, Any]], p1_idx, p3_idx, p5_idx) -> bool:
    # ... (código idêntico)
    try:
        indices = {p['idx']: i for i, p in enumerate(pivots)}
        idx_p1, idx_p3, idx_p5 = indices.get(p1_idx), indices.get(p3_idx), indices.get(p5_idx)
        if any(i is None for i in [idx_p1, idx_p3, idx_p5]) or idx_p1 < 2: return False
        p0_idx, p2_idx, p4_idx = pivots[idx_p1-1]['idx'], pivots[idx_p3-1]['idx'], pivots[idx_p5-1]['idx']
        vol_cabeca = df.loc[p2_idx:p3_idx]['volume'].mean()
        vol_od = df.loc[p4_idx:p5_idx]['volume'].mean()
        return vol_cabeca > vol_od
    except Exception: return False
    return False

def validate_and_score_hns_pattern(p0, p1, p2, p3, p4, p5, tipo_padrao, df_historico, pivots, avg_pivot_dist_days):
    """
    Valida um potencial padrão H&S, aplicando regras obrigatórias primeiro.
    Se o padrão passar, calcula um score ponderado e o retorna se atingir o mínimo.
    """
    details = {key: False for key in Config.SCORE_WEIGHTS.keys()}
    ombro_esq, neckline1, cabeca, neckline2, ombro_dir = p1, p2, p3, p4, p5

    # --- 1. Cálculos preliminares ---
    altura_cabeca = abs(cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_esq = abs(ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_dir = abs(ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))

    # --- 2. Verificações Obrigatórias ---
    # Se qualquer uma falhar, o padrão é descartado imediatamente.
    
    # Regra 1: A cabeça deve ser o extremo do padrão (pico mais alto ou vale mais baixo)
    details['valid_extremo_cabeca'] = (tipo_padrao == 'OCO' and cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']) or \
                                      (tipo_padrao == 'OCOI' and cabeca['preco'] < ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco'])
    if not details['valid_extremo_cabeca']: return None

    # Regra 2: A cabeça deve ser um extremo no contexto de um período maior
    details['valid_contexto_cabeca'] = is_head_extreme(df_historico, cabeca, avg_pivot_dist_days)
    if not details['valid_contexto_cabeca']: return None

    # Regra 3: Ombros devem ter altura simétrica (dentro de uma tolerância)
    details['valid_simetria_ombros'] = altura_cabeca > 0 and \
                                       abs(altura_ombro_esq - altura_ombro_dir) <= altura_cabeca * Config.SHOULDER_SYMMETRY_TOLERANCE
    if not details['valid_simetria_ombros']: return None

    # Regra 4: Neckline deve ser relativamente plana
    details['valid_neckline_plana'] = altura_ombro_esq > 0 and \
                                      abs(neckline1['preco'] - neckline2['preco']) <= altura_ombro_esq * Config.NECKLINE_FLATNESS_TOLERANCE
    if not details['valid_neckline_plana']: return None

    # Regra 5: O padrão deve emergir de uma tendência anterior clara
    details['valid_base_tendencia'] = (tipo_padrao == 'OCO' and p0['preco'] < neckline1['preco'] and p0['preco'] < neckline2['preco']) or \
                                      (tipo_padrao == 'OCOI' and p0['preco'] > neckline1['preco'] and p0['preco'] > neckline2['preco'])
    if not details['valid_base_tendencia']: return None

    # --- 3. Pontuação (se todas as regras obrigatórias passaram) ---
    score = 0
    # Soma os pontos das regras que já sabemos que passaram
    for rule, passed in details.items():
        if passed:
            score += Config.SCORE_WEIGHTS.get(rule, 0)

    # --- 4. Verificações Opcionais (adicionam mais pontos) ---
    
    # Proeminência da cabeça
    if altura_ombro_esq > 0 and (altura_cabeca / altura_ombro_esq >= Config.HEAD_SIGNIFICANCE_RATIO):
        details['valid_proeminencia_cabeca'] = True
        score += Config.SCORE_WEIGHTS['valid_proeminencia_cabeca']

    # Divergência de RSI
    if check_rsi_divergence(df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], tipo_padrao):
        details['valid_divergencia_rsi'] = True
        score += Config.SCORE_WEIGHTS['valid_divergencia_rsi']
        
    # Divergência de MACD
    if check_macd_divergence(df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], tipo_padrao):
        details['valid_divergencia_macd'] = True
        score += Config.SCORE_WEIGHTS['valid_divergencia_macd']

    # Fraqueza do ombro direito
    if (tipo_padrao == 'OCO' and ombro_dir['preco'] < ombro_esq['preco']) or \
       (tipo_padrao == 'OCOI' and ombro_dir['preco'] > ombro_esq['preco']):
        details['valid_ombro_direito_fraco'] = True
        score += Config.SCORE_WEIGHTS['valid_ombro_direito_fraco']
        
    # Perfil de Volume
    if check_volume_profile(df_historico, pivots, p1['idx'], p3['idx'], p5['idx']):
        details['valid_perfil_volume'] = True
        score += Config.SCORE_WEIGHTS['valid_perfil_volume']

    # --- 5. Decisão Final ---
    if score >= Config.MINIMUM_SCORE_TO_SAVE:
        base_data = {
            'padrao_tipo': tipo_padrao, 'score_total': score,
            'p0_idx': p0['idx'], 
            'ombro1_idx': p1['idx'], 'ombro1_preco': p1['preco'], 
            'neckline1_idx': p2['idx'], 'neckline1_preco': p2['preco'], 
            'cabeca_idx': p3['idx'], 'cabeca_preco': p3['preco'], 
            'neckline2_idx': p4['idx'], 'neckline2_preco': p4['preco'], 
            'ombro2_idx': p5['idx'], 'ombro2_preco': p5['preco']
        }
        # Adiciona o boletim detalhado ao resultado
        base_data.update(details)
        return base_data
    
    return None

def identificar_padroes_hns(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    padroes_encontrados = []
    n = len(pivots)
    if n < 7: return []
    
    avg_pivot_dist_days = np.mean([(pivots[i]['idx'] - pivots[i-1]['idx']).days for i in range(1, n)]) if n > 1 else 0
    
    for i in range(n - 6):
        janela = pivots[i:i+7]
        p0, p1, p2, p3, p4, p5 = janela[0], janela[1], janela[2], janela[3], janela[4], janela[5]
        
        tipo_padrao = None
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])): tipo_padrao = 'OCO'
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])): tipo_padrao = 'OCOI'
            
        if tipo_padrao:
            dados_padrao = validate_and_score_hns_pattern(p0, p1, p2, p3, p4, p5, tipo_padrao, df_historico, pivots, avg_pivot_dist_days)
            if dados_padrao:
                padroes_encontrados.append(dados_padrao)
    return padroes_encontrados

def main():
    print(f"{Style.BRIGHT}--- INICIANDO MOTOR DE GERAÇÃO (v19 - Regras Obrigatórias) ---")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    todos_os_padroes_finais = []
    
    for ticker in Config.TICKERS:
        for interval in Config.INTERVALS:
            print(f"\n--- Processando: {ticker} | Intervalo: {interval} ---")
            try:
                params = Config.TIMEFRAME_PARAMS.get(interval, Config.TIMEFRAME_PARAMS['default'])
                df_historico = buscar_dados(ticker, Config.DATA_PERIOD, interval)
                
                print(f"Calculando ZigZag com depth={params['depth']}, deviation={params['deviation']}%...")
                pivots_detectados = calcular_zigzag_oficial(df_historico, params['depth'], params['deviation'])
                
                if len(pivots_detectados) < 7:
                    print("ℹ️ Número insuficiente de pivôs para formar um padrão.")
                    continue
                
                print("Identificando padrões H&S com regras obrigatórias...")
                padroes_encontrados = identificar_padroes_hns(pivots_detectados, df_historico)

                if padroes_encontrados:
                    print(f"{Fore.GREEN}✅ Encontrados {len(padroes_encontrados)} padrões que passaram nas regras e atingiram o score.")
                    for padrao in padroes_encontrados:
                        padrao['ticker'], padrao['timeframe'] = ticker, interval
                        todos_os_padroes_finais.append(padrao)
                else:
                    print("ℹ️ Nenhum padrão passou nos critérios obrigatórios ou atingiu a pontuação mínima.")
            except Exception as e:
                print(f"{Fore.RED}❌ Erro ao processar {ticker}/{interval}: {e}")

    print(f"\n{Style.BRIGHT}--- Processo finalizado. Salvando dataset... ---{Style.RESET_ALL}")
    
    if not todos_os_padroes_finais:
        print(f"{Fore.YELLOW}Nenhum padrão foi encontrado em todas as execuções.")
        return

    df_final = pd.DataFrame(todos_os_padroes_finais)
    df_final.drop_duplicates(subset=['ticker', 'timeframe', 'padrao_tipo', 'cabeca_idx'], inplace=True, keep='first')
    
    # Organiza as colunas para melhor leitura
    cols_info = ['ticker', 'timeframe', 'padrao_tipo', 'score_total']
    cols_validacao = sorted([col for col in df_final.columns if col.startswith('valid_')])
    cols_pontos = [col for col in df_final.columns if col.endswith(('_idx', '_preco'))]
    ordem_final = cols_info + cols_validacao + cols_pontos
    df_final = df_final.reindex(columns=ordem_final)
    
    df_final.to_csv(Config.FINAL_CSV_PATH, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\n{Fore.GREEN}✅ Dataset final com {len(df_final)} padrões únicos salvo em: {Config.FINAL_CSV_PATH}")

if __name__ == "__main__":
    main()
