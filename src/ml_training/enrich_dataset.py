# enrich_dataset.py (v2 - com features de contexto de volume)

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
from tqdm import tqdm
from typing import Dict, Any, Optional
import os

# --- Funções Utilitárias (sem alterações) ---


def encontrar_extremos(df: pd.DataFrame, window_sma: int = 2) -> Optional[list]:
    """Suaviza os preços e encontra picos e vales (extremos), agora usando colunas em minúsculas."""

    # MUDANÇA: Verifica a existência da coluna 'close' em minúsculas.
    if df.empty or 'close' not in df.columns:
        return None

    # MUDANÇA: Usa e cria colunas em minúsculas.
    df['sma_close'] = df['close'].rolling(window=window_sma).mean()
    df.dropna(inplace=True)

    if df.empty:
        return None

    precos = df['sma_close'].values

    # MUDANÇA: Usa 'close' para o cálculo da sensibilidade.
    # Também corrige o FutureWarning que aparecia no seu log.
    diff = df['close'].max() - df['close'].min()
    sensitivity = float(diff.iloc[0]) if isinstance(
        diff, pd.Series) else float(diff)
    sensitivity *= 0.01

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
    time_delta = (point_idx - p1['idx']).total_seconds() / (3600 * 24)
    return p1['preco'] + slope * time_delta

# --- Função de Recálculo Geométrico (Modificada) ---


# Função recalcular_geometria_padrao corrigida

def recalcular_geometria_padrao(df_janela: pd.DataFrame, padrao_info: pd.Series) -> Optional[Dict[str, float]]:
    """
    Recebe os dados de uma janela, re-identifica os pontos chave e calcula
    tanto as features geométricas quanto as de contexto de volume.
    """
    # --- NOVO: Bloco de Sanitização de DataFrame ---
    # Garante que o DataFrame de entrada tenha o formato esperado, resolvendo o bug.
    if df_janela.empty:
        return None

    # 1. Achata o MultiIndex, se existir.
    if isinstance(df_janela.columns, pd.MultiIndex):
        df_janela.columns = df_janela.columns.get_level_values(0)

    # 2. Converte todas as colunas para minúsculas para consistência.
    df_janela.columns = [col.lower() for col in df_janela.columns]

    # 3. Garante que as colunas essenciais são numéricas
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_janela.columns:
            df_janela[col] = pd.to_numeric(df_janela[col], errors='coerce')
    df_janela.dropna(subset=['open', 'high', 'low',
                     'close', 'volume'], inplace=True)

    if df_janela.empty:
        return None
    # --- FIM DO BLOCO DE SANITIZAÇÃO ---

    extremos = encontrar_extremos(df_janela)
    if not extremos:
        return None

    data_cabeca_alvo = pd.to_datetime(padrao_info['data_cabeca'])
    tipo_padrao = padrao_info['tipo_padrao']

    # Âncora: Encontra a cabeça no novo conjunto de extremos
    try:
        cabeca = min(extremos, key=lambda x: abs(x['idx'] - data_cabeca_alvo))
        if (tipo_padrao == 'OCO' and cabeca['tipo'] != 'PICO') or \
           (tipo_padrao == 'OCOI' and cabeca['tipo'] != 'VALE'):
            return None
    except (ValueError, KeyError):
        return None

    # Re-identifica os outros 4 pontos...
    tipo_neckline = 'VALE' if tipo_padrao == 'OCO' else 'PICO'
    tipo_ombro = 'PICO' if tipo_padrao == 'OCO' else 'VALE'
    k = extremos.index(cabeca)
    neckline_p1 = next((e for e in reversed(
        extremos[:k]) if e['tipo'] == tipo_neckline), None)
    neckline_p2 = next(
        (e for e in extremos[k+1:] if e['tipo'] == tipo_neckline), None)
    if not (neckline_p1 and neckline_p2):
        return None
    idx_neckline_p1 = extremos.index(neckline_p1)
    ombro1 = next((e for e in reversed(
        extremos[:idx_neckline_p1]) if e['tipo'] == tipo_ombro), None)
    idx_neckline_p2 = extremos.index(neckline_p2)
    ombro2 = next(
        (e for e in extremos[idx_neckline_p2+1:] if e['tipo'] == tipo_ombro), None)
    if not (ombro1 and ombro2):
        return None

    # Calcula as features geométricas...
    delta_tempo_neckline = (
        neckline_p2['idx'] - neckline_p1['idx']).total_seconds() / (3600*24)
    if delta_tempo_neckline == 0:
        return None
    neckline_slope = (neckline_p2['preco'] -
                      neckline_p1['preco']) / delta_tempo_neckline
    altura_cabeca_rel = cabeca['preco'] - get_price_on_neckline(
        cabeca['idx'], neckline_p1, neckline_p2, neckline_slope)
    altura_ombro1_rel = ombro1['preco'] - get_price_on_neckline(
        ombro1['idx'], neckline_p1, neckline_p2, neckline_slope)
    altura_ombro2_rel = ombro2['preco'] - get_price_on_neckline(
        ombro2['idx'], neckline_p1, neckline_p2, neckline_slope)
    if tipo_padrao == 'OCOI':
        altura_cabeca_rel, altura_ombro1_rel, altura_ombro2_rel = - \
            altura_cabeca_rel, -altura_ombro1_rel, -altura_ombro2_rel
    if altura_cabeca_rel <= 0 or altura_ombro1_rel < 0 or altura_ombro2_rel < 0:
        return None

    features_geo = {
        'altura_rel_cabeca': altura_cabeca_rel, 'ratio_ombro_esquerdo': altura_ombro1_rel / altura_cabeca_rel,
        'ratio_ombro_direito': altura_ombro2_rel / altura_cabeca_rel,
        'ratio_simetria_altura_ombros': min(altura_ombro1_rel, altura_ombro2_rel) / max(altura_ombro1_rel, altura_ombro2_rel) if max(altura_ombro1_rel, altura_ombro2_rel) > 0 else 1.0,
        'neckline_slope': neckline_slope
    }

    # Bloco de Cálculo de Features de Volume (agora seguro)
    vela_breakout = None
    df_pos_ombro2 = df_janela[df_janela.index > ombro2['idx']]

    # Agora `vela['close']` será sempre um float, pois as colunas foram normalizadas para minúsculas.
    for idx_vela, vela in df_pos_ombro2.head(20).iterrows():
        preco_na_neckline = get_price_on_neckline(
            idx_vela, neckline_p1, neckline_p2, neckline_slope)
        if tipo_padrao == 'OCO' and vela['close'] < preco_na_neckline:
            vela_breakout = vela
            break
        elif tipo_padrao == 'OCOI' and vela['close'] > preco_na_neckline:
            vela_breakout = vela
            break

    volume_breakout_ratio = 1.0
    if vela_breakout is not None:
        vol_durante_padrao = df_janela.loc[ombro1['idx']
            :ombro2['idx'], 'volume'].mean()
        idx_breakout = df_janela.index.get_loc(vela_breakout.name)
        df_pos_breakout = df_janela.iloc[idx_breakout +
                                         1: idx_breakout + 1 + 5]
        if not df_pos_breakout.empty and vol_durante_padrao > 0:
            vol_pos_breakout = df_pos_breakout['volume'].mean()
            volume_breakout_ratio = vol_pos_breakout / vol_durante_padrao

    features_geo['volume_breakout_ratio'] = volume_breakout_ratio
    return features_geo


def main():
    """
    Função principal que orquestra o processo de enriquecimento,
    agora pulando padrões cujos dados históricos não estão mais disponíveis.
    """
    # AINDA USAMOS O dataset_corrected_final.csv que já tem a duracao_em_velas corrigida
    ARQUIVO_ENTRADA = 'data/datasets/filtered/dataset_corrected_final.csv'
    ARQUIVO_SAIDA = 'data/datasets/enriched/dataset_final_ml.csv'

    print(f"--- Iniciando Pipeline de Engenharia de Features (Geometria + Volume) ---")
    try:
        df = pd.read_csv(ARQUIVO_ENTRADA)
        df['data_inicio'] = pd.to_datetime(df['data_inicio'])
        df['data_fim'] = pd.to_datetime(df['data_fim'])
        df['data_cabeca'] = pd.to_datetime(df['data_cabeca'])
        print(
            f"Dataset '{ARQUIVO_ENTRADA}' carregado. Total de {len(df)} padrões.")
    except FileNotFoundError:
        print(f"🚨 ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
        return

    novas_colunas = ['altura_rel_cabeca', 'ratio_ombro_esquerdo', 'ratio_ombro_direito',
                     'ratio_simetria_altura_ombros', 'neckline_slope', 'volume_breakout_ratio']
    for col in novas_colunas:
        df[col] = np.nan

    print("\nIterando sobre cada padrão para calcular as features...")
    for index, linha in tqdm(df.iterrows(), total=df.shape[0], desc="Enriquecendo Padrões"):
        try:
            buffer = pd.Timedelta(days=30)
            data_inicio_busca = linha['data_inicio'] - buffer
            data_fim_busca = linha['data_fim'] + buffer

            # Tentamos buscar os dados com o intervalo original e auto_adjust=False para ter a coluna 'Volume'
            df_janela = yf.download(
                tickers=linha['ticker'], start=data_inicio_busca, end=data_fim_busca,
                interval=linha['intervalo'], progress=False, auto_adjust=False
            )

            # Se o download falhar ou retornar um DataFrame vazio, pulamos este padrão
            if df_janela.empty:
                print(
                    f"  -> AVISO [Índice {index}]: Dados para {linha['ticker']} no intervalo '{linha['intervalo']}' não disponíveis para este período. Pulando.")
                continue  # Pula para a próxima iteração do loop

            # A sanitização do índice é importante para consistência
            df_janela.index = pd.to_datetime(df_janela.index).tz_localize(None)

            features_calculadas = recalcular_geometria_padrao(df_janela, linha)

            if features_calculadas:
                for feature_nome, feature_valor in features_calculadas.items():
                    df.loc[index, feature_nome] = feature_valor

        except Exception as e:
            # Captura qualquer outro erro inesperado durante o processamento de uma linha
            print(
                f"  -> ERRO [Índice {index}]: Falha inesperada ao processar {linha['ticker']}. Erro: {e}. Pulando.")
            continue

    print("\n--- Processo de Enriquecimento Concluído ---")
    sucesso_total = df['altura_rel_cabeca'].notna().sum()
    total = len(df)
    print(
        f"Sucesso: {sucesso_total} de {total} padrões foram enriquecidos com sucesso ({sucesso_total/total:.2%}).")
    print(f"Os padrões restantes foram ignorados por falta de dados históricos na resolução necessária.")

    output_dir = os.path.dirname(ARQUIVO_SAIDA)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(ARQUIVO_SAIDA, index=False, float_format='%.4f')
    print(f"\n✅ Dataset final salvo em '{ARQUIVO_SAIDA}'.")


if __name__ == "__main__":
    main()
