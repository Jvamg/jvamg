#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de análise de padrões OCO/OCOI para ser utilizado por um screener.

Este script contém as funções para baixar dados, encontrar extremos,
calcular features e avaliar a qualidade de um padrão gráfico usando um
modelo de Machine Learning. Ele é projetado para ser importado e não
executado diretamente.
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
LIMIAR_CONFIANCA = 0.5
MODELO_PATH = 'data/models/modelo_qualidade_pattens.joblib'
PERIODO_DOWNLOAD_MAP = {
    '1h': '15d', '4h': '1mo',
    '1d': '400d', '1wk': '400wk'
}

# --- Constantes de Cor (para serem usadas pelo screener) ---
C_GREEN = Fore.GREEN
C_YELLOW = Fore.YELLOW
C_RED = Fore.RED
C_CYAN = Fore.CYAN
C_BOLD = Style.BRIGHT

# --- Funções de Apoio (Feature Engineering) ---


def encontrar_extremos(df: pd.DataFrame, window_sma: int = 2):
    """Encontra picos e vales nos dados de preço usando uma SMA para suavização."""
    if df.empty or 'close' not in df.columns:
        return []

    df_copy = df.copy()
    df_copy['sma_close'] = df_copy['close'].rolling(window=window_sma).mean()
    df_copy.dropna(inplace=True)

    if df_copy.empty:
        return []

    precos = df_copy['sma_close'].values
    sensitivity = (df_copy['close'].max() - df_copy['close'].min()) * 0.015

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


def calcular_features_padrao(ultimos_5_extremos: list, df_janela: pd.DataFrame):
    """Calcula as 16 features para um padrão candidato de 5 pontos."""
    try:
        ombro1, neckline_p1, cabeca, neckline_p2, ombro2 = ultimos_5_extremos

        delta_tempo_neckline = (
            neckline_p2['idx'] - neckline_p1['idx']).total_seconds() / (3600 * 24)
        if delta_tempo_neckline == 0:
            return None

        neckline_slope = (
            neckline_p2['preco'] - neckline_p1['preco']) / delta_tempo_neckline

        altura_cabeca_rel = abs(cabeca['preco'] - get_price_on_neckline(
            cabeca['idx'], neckline_p1, neckline_p2, neckline_slope))
        altura_ombro1_rel = abs(ombro1['preco'] - get_price_on_neckline(
            ombro1['idx'], neckline_p1, neckline_p2, neckline_slope))
        altura_ombro2_rel = abs(ombro2['preco'] - get_price_on_neckline(
            ombro2['idx'], neckline_p1, neckline_p2, neckline_slope))

        if altura_cabeca_rel == 0:
            return None

        df_padrao = df_janela.loc[ombro1['idx']:ombro2['idx']]
        vol_medio_padrao = df_padrao['volume'].mean()
        vela_ombro2 = df_padrao.loc[df_padrao.index.get_loc(
            ombro2['idx'], method='nearest')]

        volume_breakout_ratio = vela_ombro2['volume'] / \
            vol_medio_padrao if vol_medio_padrao > 0 else 1.0
        ruido_padrao = df_padrao['close'].pct_change().std()

        features = {
            'altura_rel_cabeca': altura_cabeca_rel,
            'ratio_ombro_esquerdo': altura_ombro1_rel / altura_cabeca_rel,
            'ratio_ombro_direito': altura_ombro2_rel / altura_cabeca_rel,
            'ratio_simetria_altura_ombros': min(altura_ombro1_rel, altura_ombro2_rel) / max(altura_ombro1_rel, altura_ombro2_rel) if max(altura_ombro1_rel, altura_ombro2_rel) > 0 else 1.0,
            'neckline_slope': neckline_slope,
            'volume_breakout_ratio': volume_breakout_ratio,
            'intervalo_em_minutos': (df_janela.index[1] - df_janela.index[0]).total_seconds() / 60,
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
    Analisa um único ativo e retorna uma lista de resultados encontrados.
    Esta versão é otimizada para ser chamada por outro script, recebendo os
    artefatos de ML como argumentos para evitar recarregá-los.
    """
    resultados = []

    for timeframe in TIMEFRAMES:
        # a. Download dos Dados
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
        except Exception:
            # Ignora erros de download para um ticker/timeframe específico
            continue

        # b. Encontrar Extremos
        extremos = encontrar_extremos(df)
        if len(extremos) < 5:
            continue

        # c. Identificar Candidato Recente
        ultimos_5_extremos = extremos[-5:]
        tipos_extremos = [e['tipo'] for e in ultimos_5_extremos]

        tipo_padrao = None
        if tipos_extremos == ['PICO', 'VALE', 'PICO', 'VALE', 'PICO']:
            tipo_padrao = 'OCO'  # Ombro-Cabeça-Ombro
        elif tipos_extremos == ['VALE', 'PICO', 'VALE', 'PICO', 'VALE']:
            tipo_padrao = 'OCOI'  # Ombro-Cabeça-Ombro Invertido

        if not tipo_padrao:
            continue

        # d. Calcular Features
        features = calcular_features_padrao(ultimos_5_extremos, df)
        if features is None:
            continue

        # e. Fazer a Previsão
        try:
            # Garante a ordem correta das features
            feature_values = [features[fname] for fname in expected_features]

            # Normaliza os dados
            dados_normalizados = scaler.transform(
                np.array(feature_values).reshape(1, -1))

            # Prevê a probabilidade
            probabilidades = model.predict_proba(dados_normalizados)
            confianca_valid = probabilidades[0][1]

            # f. Adicionar resultado à lista se a confiança for alta
            if confianca_valid >= LIMIAR_CONFIANCA:
                resultados.append({
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'padrao': tipo_padrao,
                    'confianca': confianca_valid
                })
        except (KeyError, ValueError):
            # Ignora erros se uma feature estiver faltando ou houver erro de valor
            continue

    # Retorna a lista de dicionários com os alertas encontrados para este ticker
    return resultados
