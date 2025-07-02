#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de análise de padrões OCO/OCOI para ser utilizado por um screener.

Este script contém as funções para baixar dados, encontrar extremos,
calcular features e avaliar a qualidade de um padrão gráfico usando um
modelo de Machine Learning. Ele é projetado para escanear um período
de dados e retornar apenas os padrões de alta confiança que se
formaram recentemente.

Versão Sincronizada: A lógica de cálculo de features foi alinhada
com a pipeline de treinamento para eliminar o "training-serving skew".
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks
from colorama import Fore, Style

# Ignorar avisos futuros do yfinance e pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuração (pode ser importada por outros scripts) ---
TIMEFRAMES = ['1h', '4h', '1d', '1wk']
LIMIAR_CONFIANCA = 0.01
MODELO_PATH = 'data\models\modelo_qualidade_pattens.joblib'
PERIODO_DOWNLOAD_MAP = {
    '1h': '15d', '4h': '1mo',
    '1d': '400d', '1wk': '400wk'
}
NUM_VELAS_RECENTES = 10

# --- Constantes de Lógica de Feature (CRÍTICO: Devem ser idênticas ao treino) ---
SENSITIVITY_FACTOR = 0.01
MAX_VELAS_BUSCA_ROMPIMENTO = 20
VELAS_VOLUME_POS_ROMPIMENTO = 5

# --- Constantes de Cor (para serem usadas pelo screener) ---
C_GREEN = Fore.GREEN
C_YELLOW = Fore.YELLOW
C_RED = Fore.RED
C_CYAN = Fore.CYAN
C_BOLD = Style.BRIGHT

# --- Funções de Apoio (Feature Engineering) Sincronizadas ---


def encontrar_extremos(df: pd.DataFrame, window_sma: int = 2):
    """Encontra picos e vales nos dados de preço usando uma SMA para suavização (sincronizado)."""
    if df.empty or 'close' not in df.columns:
        return []

    df_copy = df.copy()
    df_copy['sma_close'] = df_copy['close'].rolling(window=window_sma).mean()
    df_copy.dropna(inplace=True)

    if df_copy.empty:
        return []

    precos = df_copy['sma_close'].values
    # Sincronizado: O fator de sensibilidade é fixo para corresponder ao treino.
    sensitivity = (df_copy['close'].max() -
                   df_copy['close'].min()) * SENSITIVITY_FACTOR

    if sensitivity == 0:
        return []

    indices_picos, _ = find_peaks(precos, prominence=sensitivity)
    indices_vales, _ = find_peaks(-precos, prominence=sensitivity)

    extremos = [{'idx': df_copy.index[i], 'preco': precos[i],
                 'tipo': 'PICO'} for i in indices_picos]
    extremos += [{'idx': df_copy.index[i], 'preco': precos[i],
                  'tipo': 'VALE'} for i in indices_vales]

    extremos.sort(key=lambda x: x['idx'])
    return extremos


def get_price_on_neckline(point_idx, p1, p2, slope):
    """Calcula o preço projetado na linha de pescoço (neckline)."""
    time_delta = (point_idx - p1['idx']).total_seconds() / (3600 * 24)
    return p1['preco'] + slope * time_delta


def _calcular_volume_rompimento(df: pd.DataFrame, ombro2: dict, neckline_p1: dict, neckline_slope: float, tipo_padrao: str) -> float | None:
    """
    Função auxiliar para calcular o volume médio após o rompimento da neckline.
    """
    try:
        start_loc = df.index.get_loc(ombro2['idx'])
        df_busca = df.iloc[start_loc + 1: start_loc +
                           1 + MAX_VELAS_BUSCA_ROMPIMENTO]

        if df_busca.empty:
            return None

        for idx, vela in df_busca.iterrows():
            time_delta = (idx - neckline_p1['idx']
                          ).total_seconds() / (3600 * 24)
            neckline_price_atual = neckline_p1['preco'] + \
                neckline_slope * time_delta

            rompeu = False
            if tipo_padrao == 'OCOI' and vela['close'] > neckline_price_atual:
                rompeu = True
            elif tipo_padrao == 'OCO' and vela['close'] < neckline_price_atual:
                rompeu = True

            if rompeu:
                loc_rompimento = df.index.get_loc(idx)
                df_volume = df.iloc[loc_rompimento: loc_rompimento +
                                    VELAS_VOLUME_POS_ROMPIMENTO]

                if len(df_volume) < VELAS_VOLUME_POS_ROMPIMENTO:
                    return None

                return df_volume['volume'].mean()

        return None
    except Exception:
        return None


def calcular_features_padrao(candidato_extremos: list, df_completo: pd.DataFrame, tipo_padrao: str):
    """
    Calcula as 16 features de forma sincronizada com a pipeline de treino.
    """
    try:
        ombro1, neckline_p1, cabeca, neckline_p2, ombro2 = candidato_extremos

        delta_tempo_neckline = (
            neckline_p2['idx'] - neckline_p1['idx']).total_seconds() / (3600 * 24)
        if delta_tempo_neckline == 0:
            return None

        neckline_slope = (
            neckline_p2['preco'] - neckline_p1['preco']) / delta_tempo_neckline

        altura_cabeca_rel = abs(cabeca['preco'] - get_price_on_neckline(
            cabeca['idx'], neckline_p1, neckline_p2, neckline_slope))
        if altura_cabeca_rel == 0:
            return None

        altura_ombro1_rel = abs(ombro1['preco'] - get_price_on_neckline(
            ombro1['idx'], neckline_p1, neckline_p2, neckline_slope))
        altura_ombro2_rel = abs(ombro2['preco'] - get_price_on_neckline(
            ombro2['idx'], neckline_p1, neckline_p2, neckline_slope))

        df_padrao = df_completo.loc[ombro1['idx']:ombro2['idx']]
        vol_medio_padrao = df_padrao['volume'].mean()

        vol_medio_rompimento = _calcular_volume_rompimento(
            df=df_completo, ombro2=ombro2, neckline_p1=neckline_p1,
            neckline_slope=neckline_slope, tipo_padrao=tipo_padrao
        )

        volume_breakout_ratio = vol_medio_rompimento / \
            vol_medio_padrao if vol_medio_rompimento and vol_medio_padrao > 0 else 1.0

        ruido_padrao = df_padrao['close'].pct_change().std()

        features = {
            'altura_rel_cabeca': altura_cabeca_rel,
            'ratio_ombro_esquerdo': altura_ombro1_rel / altura_cabeca_rel,
            'ratio_ombro_direito': altura_ombro2_rel / altura_cabeca_rel,
            'ratio_simetria_altura_ombros': min(altura_ombro1_rel, altura_ombro2_rel) / max(altura_ombro1_rel, altura_ombro2_rel) if max(altura_ombro1_rel, altura_ombro2_rel) > 0 else 1.0,
            'neckline_slope': neckline_slope,
            'volume_breakout_ratio': volume_breakout_ratio,
            'intervalo_em_minutos': (df_completo.index[1] - df_completo.index[0]).total_seconds() / 60,
            'duracao_em_velas': len(df_padrao),
            'dist_ombro1_cabeca': (cabeca['idx'] - ombro1['idx']).total_seconds() / (3600*24),
            'dist_cabeca_ombro2': (ombro2['idx'] - cabeca['idx']).total_seconds() / (3600*24),
            'ratio_simetria_temporal': min((cabeca['idx'] - ombro1['idx']).total_seconds(), (ombro2['idx'] - cabeca['idx']).total_seconds()) / max((cabeca['idx'] - ombro1['idx']).total_seconds(), (ombro2['idx'] - cabeca['idx']).total_seconds()) if max((cabeca['idx'] - ombro1['idx']).total_seconds(), (ombro2['idx'] - cabeca['idx']).total_seconds()) > 0 else 1.0,
            'dif_altura_ombros_rel': (altura_ombro2_rel - altura_ombro1_rel) / altura_cabeca_rel,
            'extensao_ombro1': (neckline_p1['idx'] - ombro1['idx']).total_seconds() / (3600*24),
            'extensao_ombro2': (ombro2['idx'] - neckline_p2['idx']).total_seconds() / (3600*24),
            'neckline_angle_rad': np.arctan(neckline_slope),
            'ruido_padrao': ruido_padrao if pd.notna(ruido_padrao) else 0.0,
        }
        return features
    except Exception:
        return None

# --- Função Principal Refatorada ---


def analisar_ativo_para_screener(ticker: str, model, scaler, expected_features):
    """
    Analisa um único ativo, buscando por padrões OCO/OCOI que se completaram
    recentemente, com a lógica de features 100% sincronizada com o treino.
    """
    resultados = []

    for timeframe in TIMEFRAMES:
        try:
            df = yf.download(
                tickers=ticker,
                interval=timeframe,
                period=PERIODO_DOWNLOAD_MAP[timeframe],
                progress=False,
                auto_adjust=True
            )
            if df.empty:
                continue

            # Padroniza os nomes das colunas para minúsculas para robustez
            df.columns = [col.lower() for col in df.columns]

        except Exception:
            continue

        extremos = encontrar_extremos(df)
        if len(extremos) < 5:
            continue

        total_velas = len(df)

        # Implementação da Janela Deslizante
        for i in range(len(extremos) - 4):
            candidato_extremos = extremos[i:i+5]
            tipos_extremos = [e['tipo'] for e in candidato_extremos]

            tipo_padrao = None
            if tipos_extremos == ['PICO', 'VALE', 'PICO', 'VALE', 'PICO']:
                tipo_padrao = 'OCO'
            elif tipos_extremos == ['VALE', 'PICO', 'VALE', 'PICO', 'VALE']:
                tipo_padrao = 'OCOI'

            if not tipo_padrao:
                continue

            # Passa o `tipo_padrao` para a função de cálculo de features
            features = calcular_features_padrao(
                candidato_extremos, df, tipo_padrao)
            if features is None:
                continue

            try:
                feature_values = [features[fname]
                                  for fname in expected_features]
                dados_normalizados = scaler.transform(
                    np.array(feature_values).reshape(1, -1))
                probabilidades = model.predict_proba(dados_normalizados)
                confianca_valid = probabilidades[0][1]

                if confianca_valid >= LIMIAR_CONFIANCA:
                    # Filtro de Recência: só considera padrões recentes
                    ombro2 = candidato_extremos[4]
                    indice_fim_padrao = df.index.get_loc(ombro2['idx'])

                    if (total_velas - 1 - indice_fim_padrao) <= NUM_VELAS_RECENTES:
                        resultados.append({
                            'ticker': ticker,
                            'timeframe': timeframe,
                            'padrao': tipo_padrao,
                            'confianca': f"{confianca_valid:.2%}",
                            'data_fim': ombro2['idx'].strftime('%Y-%m-%d %H:%M')
                        })
            except (KeyError, ValueError):
                continue

    return resultados
