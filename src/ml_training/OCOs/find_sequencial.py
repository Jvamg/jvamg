# --- MOTOR DE GERAÇÃO DE DATASET DE PADRÕES HNS (v3 - COM FEATURE ENGINEERING - BUSCA SEQUENCIAL) ---
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional
import glob
import os

# --- PASSO 1: CONFIGURAÇÃO CENTRALIZADA ---
LISTA_TICKERS = [
    # Camada 1 (Alta Cap)
    'BTC-USD', 'ETH-USD',
    # Camada 2 (Plataformas Contrato Inteligente)
    'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD',
    # Camada 3 (Outros Setores - DeFi, Oráculos, etc.)
    'LINK-USD', 'UNI-USD', 'LTC-USD', 'BCH-USD', 'TRX-USD', 'SHIB-USD'
]
LISTA_INTERVALOS = ['1h', '4h', '1d', '1wk', '1mo']
PERIODO_BUSCA_PADRAO = '8y'

# --- PARÂMETROS GLOBAIS DO ALGORITMO ---
PEAK_DETECTION_SENSITIVITY = 0.015
WINDOW_SIZE = 60
HEAD_SHOULDER_RATIO_THRESHOLD = 0.92
SYMMETRY_THRESHOLD = 0.80
NECKLINE_SLOPE_TOLERANCE = 500.0
WINDOW_SMA = 2

# --- FUNÇÕES PRINCIPAIS ---


def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Busca dados históricos de um ativo usando a API do yfinance."""
    print(
        f"Buscando dados históricos de {ticker} para o período {period} e intervalo {interval}...")
    df = yf.download(tickers=ticker, period=period,
                     interval=interval, auto_adjust=True, progress=False)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ConnectionError(
            f"Não foi possível buscar os dados para {ticker} no intervalo {interval}.")
    print("Dados carregados com sucesso.")
    return df


def encontrar_extremos(df: pd.DataFrame, window_sma: int, sensitivity: float) -> List[Dict[str, Any]]:
    """Suaviza os preços e encontra picos e vales (extremos)."""
    df['SMA_Close'] = df['Close'].rolling(window=window_sma).mean()
    df.dropna(inplace=True)
    precos = df['SMA_Close'].values
    if len(precos) == 0:
        return []
    precos = np.asarray(precos)  # Garante que é um np.ndarray
    prominence_calculada = float(df['Close'].mean()) * sensitivity
    indices_picos, _ = find_peaks(precos, prominence=prominence_calculada)
    indices_vales, _ = find_peaks(-precos, prominence=prominence_calculada)
    extremos = [
        {'idx': i, 'data': df.index[i], 'preco': precos[i], 'tipo': 'PICO'} for i in indices_picos
    ] + [
        {'idx': i, 'data': df.index[i], 'preco': precos[i], 'tipo': 'VALE'} for i in indices_vales
    ]
    extremos.sort(key=lambda x: x['idx'])
    return extremos


def get_price_on_neckline(point_idx: int, p1: Dict, p2: Dict, slope: float) -> float:
    """Calcula o preço projetado na linha de pescoço para um dado ponto no tempo."""
    return p1['preco'] + slope * (point_idx - p1['idx'])


# NOVO: Função de validação unificada para padrões HNS (OCO e OCOI)
def _validate_hns_pattern(p0: Dict,ombro1: Dict, neckline_p1: Dict, cabeca: Dict, neckline_p2: Dict, ombro2: Dict, p6: Dict, tipo_padrao: str) -> Optional[Dict[str, Any]]:
    """
    Valida um conjunto de 7 pontos (p0, ombro1, neckline_p1, cabeca, neckline_p2, ombro2, p6)
    para formar um padrão OCO ou OCOI, aplicando as regras de validação.
    """
    # 1. Ordem dos pontos (já garantida pela busca sequencial, mas bom para robustez)
    if not (p0['idx'] < ombro1['idx'] < neckline_p1['idx'] < cabeca['idx'] < neckline_p2['idx'] < ombro2['idx'] < p6['idx']):
        return None

    # 2. Duração do padrão (distância entre os ombros)
    if (ombro2['idx'] - ombro1['idx']) > WINDOW_SIZE:
        return None

    # 3. Pontos da linha de pescoço devem ser distintos
    if neckline_p2['idx'] == neckline_p1['idx']:
        return None

    # 4. Inclinação da linha de pescoço
    neckline_slope = (neckline_p2['preco'] - neckline_p1['preco']) / (neckline_p2['idx'] - neckline_p1['idx'])
    if abs(neckline_slope) > NECKLINE_SLOPE_TOLERANCE:
        return None

    # 5. Alturas relativas dos pontos em relação à linha de pescoço
    altura_cabeca_rel = cabeca['preco'] - get_price_on_neckline(cabeca['idx'], neckline_p1, neckline_p2, neckline_slope)
    altura_ombro1_rel = ombro1['preco'] - get_price_on_neckline(ombro1['idx'], neckline_p1, neckline_p2, neckline_slope)
    altura_ombro2_rel = ombro2['preco'] - get_price_on_neckline(ombro2['idx'], neckline_p1, neckline_p2, neckline_slope)

    if tipo_padrao == 'OCO':
        # Para OCO (padrão de topo):
        # Ombros devem ser mais baixos que a cabeça
        if ombro1['preco'] >= cabeca['preco'] or ombro2['preco'] >= cabeca['preco']:
            return None
        # Todas as alturas relativas (acima da neckline) devem ser positivas
        if altura_cabeca_rel <= 0 or altura_ombro1_rel <= 0 or altura_ombro2_rel <= 0:
            return None
    elif tipo_padrao == 'OCOI':
        # Para OCOI (padrão de fundo):
        # Ombros devem ser mais altos que a cabeça
        if ombro1['preco'] <= cabeca['preco'] or ombro2['preco'] <= cabeca['preco']:
            return None
        # Inverte as alturas relativas para que fiquem positivas (abaixo da neckline)
        altura_cabeca_rel, altura_ombro1_rel, altura_ombro2_rel = -altura_cabeca_rel, -altura_ombro1_rel, -altura_ombro2_rel
        # Todas as alturas relativas (após inversão) devem ser positivas
        if altura_cabeca_rel <= 0 or altura_ombro1_rel <= 0 or altura_ombro2_rel <= 0:
            return None
    else:
        return None # Tipo de padrão inválido

    # 6. Proporção das alturas dos ombros em relação à cabeça
    ratio1 = altura_ombro1_rel / altura_cabeca_rel
    ratio2 = altura_ombro2_rel / altura_cabeca_rel
    if ratio1 >= HEAD_SHOULDER_RATIO_THRESHOLD or ratio2 >= HEAD_SHOULDER_RATIO_THRESHOLD:
        return None

    # 7. Simetria das alturas dos ombros
    if min(altura_ombro1_rel, altura_ombro2_rel) / max(altura_ombro1_rel, altura_ombro2_rel) < SYMMETRY_THRESHOLD:
        return None

    # 8. Simetria dos preços da linha de pescoço (pontos devem estar em nível similar)
    if min(neckline_p1['preco'], neckline_p2['preco']) / max(neckline_p1['preco'], neckline_p2['preco']) < SYMMETRY_THRESHOLD:
        return None

    # Se todas as validações passarem, retorna os detalhes do padrão
    return {
        'tipo_padrao': tipo_padrao,
        'data_inicio': p0['data'],
        'data_fim': p6['data'],
        'data_ombro1': ombro1['data'],
        'data_ombro2': ombro2['data'],
        'preco_cabeca': cabeca['preco'],
        'data_cabeca': cabeca['data'],
        'idx_inicio': p0['idx'],
        'idx_fim': p6['idx']
    }


# MODIFICADO: Agora busca sequências específicas de 7 pontos
def identificar_padroes_hns_sequential(extremos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lista_de_padroes = []
    n = len(extremos)

    # Precisamos de pelo menos 7 pontos para formar a sequência
    # V1, LS, NL1, H, NL2, RS, V4 (para OCO)
    # P1, LS, NL1, H, NL2, RS, P4 (para OCOI)
    for i in range(n - 6):
        # Extrai os 7 pontos candidatos da sequência
        p0 = extremos[i]
        p1 = extremos[i+1] # Ombro Esquerdo (LS)
        p2 = extremos[i+2] # Linha de Pescoço 1 (NL1)
        p3 = extremos[i+3] # Cabeça (H)
        p4 = extremos[i+4] # Linha de Pescoço 2 (NL2)
        p5 = extremos[i+5] # Ombro Direito (RS)
        p6 = extremos[i+6]

        # --- Tenta identificar OCO (Head and Shoulders) ---
        # Sequência de tipos esperada: VALE, PICO, VALE, PICO, VALE, PICO, VALE
        if (p0['tipo'] == 'VALE' and
            p1['tipo'] == 'PICO' and
            p2['tipo'] == 'VALE' and
            p3['tipo'] == 'PICO' and
            p4['tipo'] == 'VALE' and
            p5['tipo'] == 'PICO' and
            p6['tipo'] == 'VALE'):

            # Valida os 5 pontos centrais do padrão OCO
            padrao = _validate_hns_pattern(p0, p1, p2, p3, p4, p5, p6, 'OCO')
            if padrao:
                # Adiciona o padrão encontrado se ele ainda não estiver na lista
                if padrao not in lista_de_padroes:
                    lista_de_padroes.append(padrao)

        # --- Tenta identificar OCOI (Inverse Head and Shoulders) ---
        # Sequência de tipos esperada: PICO, VALE, PICO, VALE, PICO, VALE, PICO
        if (p0['tipo'] == 'PICO' and
            p1['tipo'] == 'VALE' and
            p2['tipo'] == 'PICO' and
            p3['tipo'] == 'VALE' and
            p4['tipo'] == 'PICO' and
            p5['tipo'] == 'VALE' and
            p6['tipo'] == 'PICO'):

            # Valida os 5 pontos centrais do padrão OCOI
            padrao = _validate_hns_pattern(p0, p1, p2, p3, p4, p5, p6, 'OCOI')
            if padrao:
                # Adiciona o padrão encontrado se ele ainda não estiver na lista
                if padrao not in lista_de_padroes:
                    lista_de_padroes.append(padrao)

    return lista_de_padroes


def salvar_resultados(df_resultados: pd.DataFrame, ticker: str, intervalo: str):
    """Salva os resultados em um CSV com nome dinâmico."""
    # NOVO: Cria um subdiretório específico para os resultados sequenciais
    output_dir = "data/datasets/padroes_hns_sequential"
    os.makedirs(output_dir, exist_ok=True) # Garante que o diretório exista

    if not df_resultados.empty:
        nome_arquivo = os.path.join(output_dir, f"padroes_hns_{ticker}_{intervalo}.csv")
        df_resultados.sort_values(
            by='data_inicio', ascending=False, inplace=True)
        df_resultados.to_csv(nome_arquivo, index=False,
                             float_format='%.2f', date_format='%Y-%m-%d %H:%M:%S')
        print(
            f"✅ Resultados para {ticker}/{intervalo} salvos em '{nome_arquivo}'!")
    else:
        print(f"ℹ️ Nenhum padrão encontrado para {ticker}/{intervalo}.")


# MUDANÇA: A função de agregação foi renomeada e agora também gera as features.
def agregar_e_gerar_features(pasta_output: str = '.') -> None:
    """Encontra, carrega, enriquece, gera features e concatena todos os CSVs."""
    print("\n--- Iniciando Estágio 2: Agregação e Engenharia de Features (Busca Sequencial) ---")
    # NOVO: Busca arquivos no novo subdiretório
    padrao_arquivo = os.path.join(
        pasta_output, "data/datasets/padroes_hns_sequential/padroes_hns_*.csv")
    arquivos_csv = glob.glob(padrao_arquivo)

    if not arquivos_csv:
        print("Nenhum arquivo de resultado encontrado para agregar na busca sequencial.")
        return

    lista_dfs = []
    for arquivo in arquivos_csv:
        try:
            nome_base = os.path.basename(arquivo)
            partes = nome_base.replace(
                'padroes_hns_', '').replace('.csv', '').split('_')
            ticker = partes[0]
            intervalo = partes[1]
            df_temp = pd.read_csv(arquivo)
            df_temp['ticker'] = ticker
            df_temp['intervalo'] = intervalo
            lista_dfs.append(df_temp)
            print(f" -> Arquivo '{arquivo}' carregado e enriquecido.")
        except Exception as e:
            print(f" -> ⚠️ Erro ao processar o arquivo '{arquivo}': {e}")

    if not lista_dfs:
        print("Nenhum DataFrame pôde ser carregado. A agregação foi cancelada.")
        return

    df_master = pd.concat(lista_dfs, ignore_index=True)

    # NOVO: Bloco de Engenharia de Features
    print("Calculando novas features...")

    # Feature 1: Duração do padrão em velas (candles)
    df_master['duracao_em_velas'] = df_master['idx_fim'] - \
        df_master['idx_inicio']
    print("  -> Feature 'duracao_em_velas' criada.")

    # Feature 2: Mapeamento do intervalo de tempo para minutos
    mapa_intervalo_minutos = {
        '1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '90m': 90,
        '1h': 60, '4h': 240, '1d': 1440, '5d': 7200, '1wk': 10080
    }
    df_master['intervalo_em_minutos'] = df_master['intervalo'].map(
        lambda x: mapa_intervalo_minutos.get(x, np.nan))
    print("  -> Feature 'intervalo_em_minutos' criada.")

    # Reordenando colunas para melhor legibilidade do CSV
    colunas_ordem = [
        'ticker', 'intervalo', 'tipo_padrao', 'data_inicio', 'data_fim',
        'duracao_em_velas', 'intervalo_em_minutos', 'data_cabeca', 'preco_cabeca',
        'idx_inicio', 'idx_fim'
    ]
    outras_colunas = [
        col for col in df_master.columns if col not in colunas_ordem]
    df_master = df_master[colunas_ordem + outras_colunas]

    # MUDANÇA: Nome do arquivo de saída final foi alterado para refletir a busca sequencial
    nome_arquivo_final = "data/datasets/filtered/dataset_featured_sequential.csv"
    df_master.to_csv(nome_arquivo_final, index=False,
                     float_format='%.2f', date_format='%Y-%m-%d %H:%M:%S')
    print(
        f"\n🚀 Dataset com features criado com sucesso! {len(df_master)} exemplos agregados em '{nome_arquivo_final}'.")


if __name__ == "__main__":
    print("--- INICIANDO MOTOR DE GERAÇÃO DE DATASET EM LOTE (BUSCA SEQUENCIAL) ---")
    for ticker_atual in LISTA_TICKERS:
        for intervalo_atual in LISTA_INTERVALOS:
            print(
                f"\n--- Processando: {ticker_atual} | Intervalo: {intervalo_atual} ---")
            periodo_para_buscar = PERIODO_BUSCA_PADRAO
            if intervalo_atual == '4h' or intervalo_atual == '1h':
                periodo_para_buscar = '2y'
                print(
                    f"ℹ️ AJUSTE: O período de busca foi modificado para '{periodo_para_buscar}' para o intervalo '{intervalo_atual}' devido às regras da API.")
            try:
                df_historico = buscar_dados(
                    ticker_atual, periodo_para_buscar, intervalo_atual)
                extremos_detectados = encontrar_extremos(
                    df_historico, WINDOW_SMA, PEAK_DETECTION_SENSITIVITY)
                if not extremos_detectados:
                    print(
                        f"ℹ️ Nenhum extremo detectado para {ticker_atual}/{intervalo_atual}, pulando para o próximo.")
                    continue
                # MODIFICADO: Chama a nova função de identificação sequencial
                padroes_encontrados = identificar_padroes_hns_sequential(
                    extremos_detectados)
                if padroes_encontrados:
                    df_resultados = pd.DataFrame(padroes_encontrados)
                    salvar_resultados(
                        df_resultados, ticker_atual, intervalo_atual)
                else:
                    print(
                        f"ℹ️ Nenhum padrão encontrado para {ticker_atual}/{intervalo_atual} com os parâmetros atuais na busca sequencial.")
            except Exception as e:
                print(
                    f"❌ Erro ao processar o par {ticker_atual}/{intervalo_atual}: {e}")
                print("Continuando para a próxima combinação...")

    agregar_e_gerar_features()

    print("\n--- Processo em lote (Busca Sequencial) finalizado. ---")