# --- MOTOR DE GERAÇÃO DE DATASET DE PADRÕES HNS (v4 - REATORADO PARA SAÍDA BRUTA) ---
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional
import os
from colorama import Fore, Style, init

# Inicializa o Colorama
init(autoreset=True)

# --- PASSO 1: CONFIGURAÇÃO CENTRALIZADA ---
LISTA_TICKERS = ['HBAR-USD']
LISTA_INTERVALOS = ['1h', '4h', '1d']
PERIODO_BUSCA_PADRAO = '1y'

# --- PARÂMETROS GLOBAIS DO ALGORITMO (Mantidos do seu original) ---
PEAK_DETECTION_SENSITIVITY = 0.015
WINDOW_SIZE = 60
HEAD_SHOULDER_RATIO_THRESHOLD = 0.92
SYMMETRY_THRESHOLD = 0.80
NECKLINE_SLOPE_TOLERANCE = 500.0
WINDOW_SMA = 2

# Caminho para salvar o dataset final
OUTPUT_CSV_PATH = 'data/datasets/candidatos_brutos_para_predicao.csv'


# --- FUNÇÕES DE DETECÇÃO (Com pequenas adaptações) ---

# No seu script gerador, substitua a função buscar_dados inteira por esta:

def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Busca dados históricos de um ativo usando a API do yfinance.
    Versão robusta que lida com colunas de múltiplos níveis (MultiIndex).
    """
    print(f"Buscando dados históricos de {ticker} para o período {period} e intervalo {interval}...")
    
    # A chamada de download permanece a mesma
    df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    
    if df.empty:
        raise ConnectionError(f"Não foi possível buscar os dados para {ticker} no intervalo {interval}.")

    # --- INÍCIO DA CORREÇÃO ---
    # Verifica se as colunas são do tipo MultiIndex (contêm tuplas)
    if isinstance(df.columns, pd.MultiIndex):
        # Se for, "achata" o MultiIndex pegando apenas o primeiro nível dos nomes.
        # Ex: ('Close', '') se torna 'Close'
        df.columns = df.columns.get_level_values(0)
    # --- FIM DA CORREÇÃO ---

    # Agora que garantimos que os nomes das colunas são strings, esta linha funcionará
    df.columns = [col.lower() for col in df.columns]
    
    print("Dados carregados com sucesso.")
    return df

def encontrar_extremos(df: pd.DataFrame, window_sma: int, sensitivity: float) -> List[Dict[str, Any]]:
    """Suaviza os preços e encontra picos e vales (extremos)."""
    df['sma_close'] = df['close'].rolling(window=window_sma).mean()
    df.dropna(inplace=True)
    precos = df['sma_close'].values
    if len(precos) == 0: return []
    
    prominence_calculada = float(df['close'].mean()) * sensitivity
    indices_picos, _ = find_peaks(precos, prominence=prominence_calculada)
    indices_vales, _ = find_peaks(-precos, prominence=prominence_calculada)
    
    # MUDANÇA: Simplificado para usar o índice do DataFrame diretamente, que é um DatetimeIndex
    extremos = [{'idx': df.index[i], 'preco': precos[i], 'tipo': 'PICO'} for i in indices_picos]
    extremos += [{'idx': df.index[i], 'preco': precos[i], 'tipo': 'VALE'} for i in indices_vales]
    
    extremos.sort(key=lambda x: x['idx'])
    return extremos

def get_price_on_neckline(point_idx, p1: Dict, p2: Dict, slope: float) -> float:
    """Calcula o preço projetado na linha de pescoço."""
    # MUDANÇA: Adaptado para usar DatetimeIndex
    time_delta_seconds = (point_idx - p1['idx']).total_seconds()
    return p1['preco'] + slope * time_delta_seconds

# MUDANÇA CRÍTICA: A função agora retorna os dados brutos necessários para o CSV
def _validate_and_extract_hns_pattern(p0: Dict, ombro1: Dict, neckline_p1: Dict, cabeca: Dict, neckline_p2: Dict, ombro2: Dict, p6: Dict, tipo_padrao: str) -> Optional[Dict[str, Any]]:
    """Valida o padrão e, se for válido, extrai os dados brutos dos 5 pontos principais."""
    # (Toda a sua lógica de validação original é mantida aqui...)
    if not (p0['idx'] < ombro1['idx'] < neckline_p1['idx'] < cabeca['idx'] < neckline_p2['idx'] < ombro2['idx'] < p6['idx']): return None
    
    time_diff_ns = (neckline_p2['idx'] - neckline_p1['idx']).value # Em nanosegundos
    if time_diff_ns == 0: return None
    
    neckline_slope = (neckline_p2['preco'] - neckline_p1['preco']) / time_diff_ns

    # (Lógica de validação de OCO/OCOI, alturas, simetria, etc. continua aqui)
    # ... (omitido para brevidade, mas deve ser mantido do seu script original) ...

    # MUDANÇA PRINCIPAL: O que a função retorna
    # Se todas as validações passarem, retorna um dicionário com os dados brutos
    return {
        'padrao_tipo': tipo_padrao,
        'ombro1_idx': ombro1['idx'], 'ombro1_preco': ombro1['preco'],
        'neckline1_idx': neckline_p1['idx'], 'neckline1_preco': neckline_p1['preco'],
        'cabeca_idx': cabeca['idx'], 'cabeca_preco': cabeca['preco'],
        'neckline2_idx': neckline_p2['idx'], 'neckline2_preco': neckline_p2['preco'],
        'ombro2_idx': ombro2['idx'], 'ombro2_preco': ombro2['preco'],
    }

def identificar_padroes_hns_sequential(extremos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifica padrões e coleta os dicionários de dados brutos."""
    lista_de_padroes_brutos = []
    n = len(extremos)
    if n < 7: return []

    for i in range(n - 6):
        p0, p1, p2, p3, p4, p5, p6 = extremos[i:i+7]
        
        tipo_padrao = None
        if all(p['tipo'] == t for p, t in zip(extremos[i:i+7], ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'OCO'
        elif all(p['tipo'] == t for p, t in zip(extremos[i:i+7], ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'OCOI'

        if tipo_padrao:
            # A função de validação agora retorna os dados que precisamos
            dados_padrao = _validate_and_extract_hns_pattern(p0, p1, p2, p3, p4, p5, p6, tipo_padrao)
            if dados_padrao:
                lista_de_padroes_brutos.append(dados_padrao)
                
    return lista_de_padroes_brutos

# --- ORQUESTRADOR PRINCIPAL (Refatorado) ---
def main():
    print(f"{Style.BRIGHT}--- INICIANDO MOTOR DE GERAÇÃO DE DATASET (v4 - SAÍDA BRUTA) ---")
    todos_os_padroes_finais = []

    for ticker_atual in LISTA_TICKERS:
        for intervalo_atual in LISTA_INTERVALOS:
            print(f"\n--- Processando: {ticker_atual} | Intervalo: {intervalo_atual} ---")
            try:
                df_historico = buscar_dados(ticker_atual, PERIODO_BUSCA_PADRAO, intervalo_atual)
                extremos_detectados = encontrar_extremos(df_historico, WINDOW_SMA, PEAK_DETECTION_SENSITIVITY)
                if not extremos_detectados:
                    print(f"ℹ️ Nenhum extremo detectado. Pulando.")
                    continue

                # A função agora retorna uma lista de dicionários com os dados brutos
                padroes_encontrados = identificar_padroes_hns_sequential(extremos_detectados)

                if padroes_encontrados:
                    print(f"{Fore.GREEN}  -> Encontrados {len(padroes_encontrados)} padrões para {ticker_atual}/{intervalo_atual}.")
                    # Adiciona o contexto de ticker e intervalo a cada padrão encontrado
                    for padrao in padroes_encontrados:
                        padrao['ticker'] = ticker_atual
                        padrao['timeframe'] = intervalo_atual
                        todos_os_padroes_finais.append(padrao)
                else:
                    print(f"ℹ️ Nenhum padrão válido encontrado.")
            
            except Exception as e:
                print(f"{Fore.RED}❌ Erro ao processar o par {ticker_atual}/{intervalo_atual}: {e}")

    # --- SALVAMENTO FINAL (Processo de estágio único) ---
    if not todos_os_padroes_finais:
        print(f"\n{Fore.YELLOW}Processo finalizado. Nenhum padrão foi encontrado em toda a análise.")
        return

    print(f"\n{Style.BRIGHT}{Fore.GREEN}Análise concluída! Total de {len(todos_os_padroes_finais)} padrões encontrados.")
    df_final = pd.DataFrame(todos_os_padroes_finais)
    
    # Garante a ordem correta das colunas para o script de predição
    ordem_colunas = [
        'ticker', 'timeframe', 'padrao_tipo',
        'ombro1_idx', 'ombro1_preco',
        'neckline1_idx', 'neckline1_preco',
        'cabeca_idx', 'cabeca_preco',
        'neckline2_idx', 'neckline2_preco',
        'ombro2_idx', 'ombro2_preco'
    ]
    df_final = df_final[ordem_colunas]

    try:
        # Garante que o diretório de saída exista
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        df_final.to_csv(OUTPUT_CSV_PATH, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Dataset bruto salvo com sucesso em: {Style.BRIGHT}{OUTPUT_CSV_PATH}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Erro ao salvar o arquivo CSV final: {e}")

if __name__ == "__main__":
    main()