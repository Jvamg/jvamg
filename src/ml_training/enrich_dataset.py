# enrich_dataset.py

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
from tqdm import tqdm  # Usaremos para uma bela barra de progresso!
from typing import Dict, Any, Optional
import os

# --- Funções Utilitárias Reutilizadas do Detector Original ---
# Estas funções são nossas ferramentas básicas de análise de geometria.

def encontrar_extremos(df: pd.DataFrame, window_sma: int = 2) -> Optional[list]:
    """Suaviza os preços e encontra picos e vales (extremos)."""
    if df.empty or 'Close' not in df.columns:
        return None
        
    # Usamos uma janela de SMA pequena para não distorcer a forma original
    df['SMA_Close'] = df['Close'].rolling(window=window_sma).mean()
    df.dropna(inplace=True)
    
    if df.empty:
        return None

    precos = df['SMA_Close'].values
    
    # Sensibilidade dinâmica para encontrar picos relevantes na janela
    # PARA:
    sensitivity = float(df['Close'].max() - df['Close'].min()) * 0.01
    indices_picos, _ = find_peaks(precos, prominence=sensitivity)
    indices_vales, _ = find_peaks(-precos, prominence=sensitivity)
    
    extremos = [
        {'idx': df.index[i], 'preco': precos[i], 'tipo': 'PICO'} for i in indices_picos
    ] + [
        {'idx': df.index[i], 'preco': precos[i], 'tipo': 'VALE'} for i in indices_vales
    ]
    
    extremos.sort(key=lambda x: x['idx'])
    return extremos

def get_price_on_neckline(point_idx: pd.Timestamp, p1: Dict, p2: Dict, slope: float) -> float:
    """Calcula o preço projetado na linha de pescoço para um dado ponto no tempo."""
    # A diferença de tempo é convertida para um valor numérico (dias) para o cálculo
    time_delta = (point_idx - p1['idx']).total_seconds() / (3600 * 24)
    return p1['preco'] + slope * time_delta

# --- O Coração do Script: A Função de Recálculo Geométrico ---

def recalcular_geometria_padrao(df_janela: pd.DataFrame, padrao_info: pd.Series) -> Optional[Dict[str, float]]:
    """
    Recebe os dados de uma janela e as informações de um padrão,
    re-identifica os pontos chave e calcula as features geométricas.
    """
    extremos = encontrar_extremos(df_janela)
    if not extremos:
        return None

    data_cabeca_alvo = pd.to_datetime(padrao_info['data_cabeca'])
    tipo_padrao = padrao_info['tipo_padrao']
    
    # 1. Âncora: Encontra a cabeça no novo conjunto de extremos
    try:
        cabeca = min(extremos, key=lambda x: abs(x['idx'] - data_cabeca_alvo))
        # Validação extra: se o extremo mais próximo não for do tipo certo, falha.
        if (tipo_padrao == 'OCO' and cabeca['tipo'] != 'PICO') or \
           (tipo_padrao == 'OCOI' and cabeca['tipo'] != 'VALE'):
           return None
    except (ValueError, KeyError):
        return None

    # 2. Re-identifica os outros 4 pontos com base na cabeça encontrada
    tipo_neckline = 'VALE' if tipo_padrao == 'OCO' else 'PICO'
    tipo_ombro = 'PICO' if tipo_padrao == 'OCO' else 'VALE'
    k = extremos.index(cabeca)

    neckline_p1 = next((e for e in reversed(extremos[:k]) if e['tipo'] == tipo_neckline), None)
    neckline_p2 = next((e for e in extremos[k+1:] if e['tipo'] == tipo_neckline), None)
    if not (neckline_p1 and neckline_p2): return None

    idx_neckline_p1 = extremos.index(neckline_p1)
    ombro1 = next((e for e in reversed(extremos[:idx_neckline_p1]) if e['tipo'] == tipo_ombro), None)
    idx_neckline_p2 = extremos.index(neckline_p2)
    ombro2 = next((e for e in extremos[idx_neckline_p2+1:] if e['tipo'] == tipo_ombro), None)
    if not (ombro1 and ombro2): return None
    
    # 3. Calcula as features geométricas
    delta_tempo_neckline = (neckline_p2['idx'] - neckline_p1['idx']).total_seconds() / (3600*24)
    if delta_tempo_neckline == 0: return None
    
    neckline_slope = (neckline_p2['preco'] - neckline_p1['preco']) / delta_tempo_neckline

    altura_cabeca_rel = cabeca['preco'] - get_price_on_neckline(cabeca['idx'], neckline_p1, neckline_p2, neckline_slope)
    altura_ombro1_rel = ombro1['preco'] - get_price_on_neckline(ombro1['idx'], neckline_p1, neckline_p2, neckline_slope)
    altura_ombro2_rel = ombro2['preco'] - get_price_on_neckline(ombro2['idx'], neckline_p1, neckline_p2, neckline_slope)
    
    # Inverte os sinais para OCOI para que as alturas sejam sempre positivas
    if tipo_padrao == 'OCOI':
        altura_cabeca_rel, altura_ombro1_rel, altura_ombro2_rel = -altura_cabeca_rel, -altura_ombro1_rel, -altura_ombro2_rel

    if altura_cabeca_rel <= 0 or altura_ombro1_rel < 0 or altura_ombro2_rel < 0: return None

    # 4. Monta o dicionário de novas features
    features = {
        'altura_rel_cabeca': altura_cabeca_rel,
        'ratio_ombro_esquerdo': altura_ombro1_rel / altura_cabeca_rel,
        'ratio_ombro_direito': altura_ombro2_rel / altura_cabeca_rel,
        'ratio_simetria_altura_ombros': min(altura_ombro1_rel, altura_ombro2_rel) / max(altura_ombro1_rel, altura_ombro2_rel) if max(altura_ombro1_rel, altura_ombro2_rel) > 0 else 1.0,
        'neckline_slope': neckline_slope
    }
    return features

def main():
    """Função principal que orquestra o processo de enriquecimento."""
    ARQUIVO_ENTRADA = 'data/datasets/filtered/dataset_labeled.csv'
    ARQUIVO_SAIDA = 'data/datasets/enriched/dataset_final_ml.csv'

    print(f"--- Iniciando Pipeline de Engenharia de Features Geométricas ---")
    try:
        df = pd.read_csv(ARQUIVO_ENTRADA)
        df['data_inicio'] = pd.to_datetime(df['data_inicio'])
        df['data_fim'] = pd.to_datetime(df['data_fim'])
        df['data_cabeca'] = pd.to_datetime(df['data_cabeca'])
        print(f"Dataset '{ARQUIVO_ENTRADA}' carregado. Total de {len(df)} padrões.")
    except FileNotFoundError:
        print(f"🚨 ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
        return

    # Inicializa as novas colunas com NaN (Not a Number)
    novas_colunas = ['altura_rel_cabeca', 'ratio_ombro_esquerdo', 'ratio_ombro_direito', 
                     'ratio_simetria_altura_ombros', 'neckline_slope']
    for col in novas_colunas:
        df[col] = np.nan

    print("\nIterando sobre cada padrão para calcular as features geométricas...")
    # Usamos tqdm para criar uma barra de progresso elegante
    for index, linha in tqdm(df.iterrows(), total=df.shape[0], desc="Enriquecendo Padrões"):
        try:
            # Buffer de tempo para garantir contexto na análise
            buffer = pd.Timedelta(days=30)
            data_inicio_busca = linha['data_inicio'] - buffer
            data_fim_busca = linha['data_fim'] + buffer

            # Busca os dados brutos para a janela do padrão
            df_janela = yf.download(
                tickers=linha['ticker'],
                start=data_inicio_busca,
                end=data_fim_busca,
                interval=linha['intervalo'],
                progress=False,
                auto_adjust=True
            )
            
            if df_janela.empty:
                print(f"  -> AVISO [Índice {index}]: Não foi possível baixar dados para {linha['ticker']}. Pulando.")
                continue

            df_janela.index = df_janela.index.tz_localize(None)  # Remove o fuso horário

            # A mágica acontece aqui: recalculamos a geometria
            features_geometricas = recalcular_geometria_padrao(df_janela, linha)

            # Se o recálculo for bem sucedido, preenchemos as colunas
            if features_geometricas:
                for feature_nome, feature_valor in features_geometricas.items():
                    df.loc[index, feature_nome] = feature_valor
        
        except Exception as e:
            print(f"  -> ERRO [Índice {index}]: Falha inesperada ao processar {linha['ticker']}. Erro: {e}. Pulando.")
            continue # Continua para a próxima linha em caso de erro

    print("\n--- Processo de Enriquecimento Concluído ---")
    
    # Análise dos resultados
    sucesso = df['altura_rel_cabeca'].notna().sum()
    total = len(df)
    print(f"Sucesso: {sucesso} de {total} padrões foram enriquecidos com sucesso ({sucesso/total:.2%}).")
    
    print(f"\nVerificando e criando diretório de saída para '{ARQUIVO_SAIDA}'...")
    output_dir = os.path.dirname(ARQUIVO_SAIDA)

    # Garante que só tentamos criar se o caminho não for a pasta atual
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Salva o dataset final, pronto para o Machine Learning
    df.to_csv(ARQUIVO_SAIDA, index=False, float_format='%.4f')
    print(f"\n✅ Dataset final salvo em '{ARQUIVO_SAIDA}'. Pronto para o treinamento!")

if __name__ == "__main__":
    main()