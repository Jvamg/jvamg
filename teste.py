# --- MOTOR DE GERAÇÃO DE DATASET OCO - v14 (RETRY DE DOWNLOAD) ---
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional
import os
import glob
import time # Importado para a lógica de espera
from colorama import Fore, Style, init

# Inicializa o Colorama
init(autoreset=True)

class Config:
    """ Parâmetros de configuração para o detector de padrões. """
    TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'HBAR-USD']
    INTERVALS = ['1h', '4h', '1d']
    DATA_PERIOD = '5y'
    
    TIMEFRAME_PARAMS = {
        'default': {'depth': 2, 'deviation': 2.0},
        '1h':      {'depth': 3, 'deviation': 4.0},
        '4h':      {'depth': 3, 'deviation': 5.0},
        '1d':      {'depth': 5, 'deviation': 8.0}
    }
    STRATEGIES = [
        {'name': 'zigzag_oficial', 'params': TIMEFRAME_PARAMS}
    ]

    SHOULDER_SYMMETRY_TOLERANCE = 0.10
    NECKLINE_FLATNESS_TOLERANCE = 0.10
    HEAD_SIGNIFICANCE_RATIO = 1.2
    HEAD_EXTREME_LOOKBACK_FACTOR = 5
    
    OUTPUT_DIR = 'data/datasets_hns_oficial'
    FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, 'dataset_hns_consolidado_final.csv')

    # --- NOVOS PARÂMETROS PARA O RETRY ---
    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 5

def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """ 
    Busca dados históricos com lógica de retry para lidar com falhas de rede.
    """
    original_period = period
    if 'm' in interval: period = '60d'
    elif 'h' in interval: period = '2y'
    if period != original_period:
        print(f"{Fore.YELLOW}Aviso: Período ajustado de '{original_period}' para '{period}' para o intervalo '{interval}'.{Style.RESET_ALL}")
    
    # --- INÍCIO DA MUDANÇA: Lógica de Retry ---
    for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
        try:
            print(f"Buscando dados para {ticker}/{interval}... Tentativa {tentativa + 1}/{Config.MAX_DOWNLOAD_TENTATIVAS}")
            df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            
            if not df.empty:
                print(f"{Fore.GREEN}Download bem-sucedido.{Style.RESET_ALL}")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [col.lower() for col in df.columns]
                return df # Retorna o DataFrame se o download for bem-sucedido
            else:
                raise ValueError("Download retornou um DataFrame vazio.")

        except Exception as e:
            print(f"{Fore.YELLOW}Falha na tentativa {tentativa + 1}: {e}{Style.RESET_ALL}")
            if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1:
                print(f"Aguardando {Config.RETRY_DELAY_SEGUNDOS} segundos para tentar novamente...")
                time.sleep(Config.RETRY_DELAY_SEGUNDOS)
            else:
                print(f"{Fore.RED}Todas as tentativas de download para {ticker}/{interval} falharam.{Style.RESET_ALL}")
                raise ConnectionError(f"Download falhou para {ticker}/{interval} após {Config.MAX_DOWNLOAD_TENTATIVAS} tentativas.")
    
    # Esta linha não deve ser alcançada, mas é um fallback
    raise ConnectionError(f"Falha inesperada no download para {ticker}/{interval}.")
    # --- FIM DA MUDANÇA ---

def calcular_zigzag_oficial(df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
    window_size = 2 * depth + 1
    rolling_max = df['high'].rolling(window=window_size, center=True, min_periods=1).max()
    rolling_min = df['low'].rolling(window=window_size, center=True, min_periods=1).min()
    
    candidate_peaks_df = df[df['high'] == rolling_max]
    candidate_valleys_df = df[df['low'] == rolling_min]
    
    candidates = []
    for idx, row in candidate_peaks_df.iterrows(): candidates.append({'idx': idx, 'preco': row['high'], 'tipo': 'PICO'})
    for idx, row in candidate_valleys_df.iterrows(): candidates.append({'idx': idx, 'preco': row['low'], 'tipo': 'VALE'})
    candidates = sorted(list({p['idx']: p for p in candidates}.values()), key=lambda x: x['idx'])
    
    if len(candidates) < 2: return []

    confirmed_pivots = [candidates[0]]
    last_pivot = candidates[0]
    for i in range(1, len(candidates)):
        candidate = candidates[i]
        if candidate['tipo'] == last_pivot['tipo']:
            if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or \
               (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                confirmed_pivots[-1] = candidate
                last_pivot = candidate
            continue
        if last_pivot['preco'] == 0: continue
        price_dev = abs(candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
        if price_dev >= deviation_percent:
            confirmed_pivots.append(candidate)
            last_pivot = candidate
    return confirmed_pivots

def is_head_extreme(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_days: int) -> bool:
    lookback_period = int(avg_pivot_dist_days * Config.HEAD_EXTREME_LOOKBACK_FACTOR)
    if lookback_period <= 0: return True
    start_date = head_pivot['idx'] - pd.Timedelta(days=lookback_period)
    end_date = head_pivot['idx'] + pd.Timedelta(days=lookback_period)
    context_df = df.loc[start_date:end_date]
    if context_df.empty: return True
    if head_pivot['tipo'] == 'PICO': return head_pivot['preco'] >= context_df['high'].max()
    else: return head_pivot['preco'] <= context_df['low'].min()

def _validate_hns_pattern(p0: Dict, p1: Dict, p2: Dict, p3: Dict, p4: Dict, p5: Dict, tipo_padrao: str, df_historico: pd.DataFrame, avg_pivot_dist: int) -> Optional[Dict[str, Any]]:
    ombro_esq, neckline1, cabeca, neckline2, ombro_dir = p1, p2, p3, p4, p5
    if tipo_padrao == 'OCO':
        if not (cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']): return None
        if not (ombro_esq['preco'] > neckline1['preco'] and ombro_dir['preco'] > neckline2['preco']): return None
        if not (ombro_dir['preco'] < ombro_esq['preco']): return None 
        if not (p0['preco'] < neckline1['preco'] and p0['preco'] < neckline2['preco']): return None
    elif tipo_padrao == 'OCOI':
        if not (cabeca['preco'] < ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco']): return None
        if not (ombro_esq['preco'] < neckline1['preco'] and ombro_dir['preco'] < neckline2['preco']): return None
        if not (ombro_dir['preco'] > ombro_esq['preco']): return None
        if not (p0['preco'] > neckline1['preco'] and p0['preco'] > neckline2['preco']): return None
    altura_cabeca = abs(cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_esq = abs(ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_dir = abs(ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    if altura_ombro_esq == 0 or altura_cabeca == 0: return None
    
    # As duas linhas a seguir estavam comentadas no seu código, mantive assim.
    # Remova o '#' se quiser reativar a validação mais rigorosa de simetria.
    #if altura_cabeca / altura_ombro_esq < Config.HEAD_SIGNIFICANCE_RATIO: return None
    #if abs(altura_ombro_esq - altura_ombro_dir) > altura_cabeca * Config.SHOULDER_SYMMETRY_TOLERANCE: return None
    
    if abs(neckline1['preco'] - neckline2['preco']) > altura_ombro_esq * Config.NECKLINE_FLATNESS_TOLERANCE: return None
    if not is_head_extreme(df_historico, cabeca, avg_pivot_dist): return None
    return {'padrao_tipo': tipo_padrao, 'ombro1_idx': ombro_esq['idx'], 'ombro1_preco': ombro_esq['preco'],'neckline1_idx': neckline1['idx'], 'neckline1_preco': neckline1['preco'],'cabeca_idx': cabeca['idx'], 'cabeca_preco': cabeca['preco'],'neckline2_idx': neckline2['idx'], 'neckline2_preco': neckline2['preco'],'ombro2_idx': ombro_dir['idx'], 'ombro2_preco': ombro_dir['preco']}

def identificar_padroes_hns(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    padroes_encontrados = []
    n = len(pivots)
    if n < 7: return []
    datas_pivots = [p['idx'] for p in pivots]
    avg_pivot_dist_days = np.mean([(datas_pivots[i] - datas_pivots[i-1]).days for i in range(1, n)]) if n > 1 else 0
    for i in range(n - 6):
        janela = pivots[i:i+7]
        p0, p1, p2, p3, p4, p5 = janela[0], janela[1], janela[2], janela[3], janela[4], janela[5]
        tipo_padrao = None
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])): tipo_padrao = 'OCO'
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])): tipo_padrao = 'OCOI'
        if tipo_padrao:
            dados_padrao = _validate_hns_pattern(p0, p1, p2, p3, p4, p5, tipo_padrao, df_historico, avg_pivot_dist_days)
            if dados_padrao:
                padroes_encontrados.append(dados_padrao)
    return padroes_encontrados

def main():
    print(f"{Style.BRIGHT}--- INICIANDO MOTOR DE GERAÇÃO (v14 - Retry) ---")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    todos_os_padroes_finais = []
    
    for strategy in Config.STRATEGIES:
        strategy_name = strategy['name']
        strategy_params = strategy['params']
        
        print(f"\n{Style.BRIGHT}{Fore.CYAN}=== EXECUTANDO ESTRATÉGIA: {strategy_name} ==={Style.RESET_ALL}")

        for ticker in Config.TICKERS:
            for interval in Config.INTERVALS:
                print(f"\n--- Processando: {ticker} | Intervalo: {interval} ---")
                try:
                    params = strategy_params.get(interval, strategy_params['default'])
                    df_historico = buscar_dados(ticker, Config.DATA_PERIOD, interval)
                    
                    print(f"Calculando ZigZag com depth={params['depth']} e deviation={params['deviation']}%...")
                    pivots_detectados = calcular_zigzag_oficial(df_historico, params['depth'], params['deviation'])
                    
                    if len(pivots_detectados) < 7:
                        print("ℹ️ Número insuficiente de pivôs para formar um padrão.")
                        continue
                    
                    padroes_encontrados = identificar_padroes_hns(pivots_detectados, df_historico)

                    if padroes_encontrados:
                        print(f"{Fore.GREEN}✅ Encontrados {len(padroes_encontrados)} padrões de alta qualidade.")
                        for padrao in padroes_encontrados:
                            padrao['ticker'], padrao['timeframe'] = ticker, interval
                            padrao['estrategia_zigzag'] = f"{strategy_name}_d{params['depth']}_dev{params['deviation']}"
                            todos_os_padroes_finais.append(padrao)
                    else:
                        print("ℹ️ Nenhum padrão com as regras rigorosas foi encontrado.")
                except Exception as e:
                    print(f"{Fore.RED}❌ Erro ao processar {ticker}/{interval}: {e}")

    print(f"\n{Style.BRIGHT}--- Processo finalizado. Salvando dataset consolidado... ---{Style.RESET_ALL}")
    
    if not todos_os_padroes_finais:
        print(f"{Fore.YELLOW}Nenhum padrão foi encontrado em todas as execuções.")
        return

    df_final = pd.DataFrame(todos_os_padroes_finais)
    subset_cols = [col for col in df_final.columns if col != 'estrategia_zigzag']
    df_final.drop_duplicates(subset=subset_cols, inplace=True, keep='first')
    
    df_final.to_csv(Config.FINAL_CSV_PATH, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\n{Fore.GREEN}✅ Dataset final com {len(df_final)} padrões únicos salvo em: {Config.FINAL_CSV_PATH}")

if __name__ == "__main__":
    main()