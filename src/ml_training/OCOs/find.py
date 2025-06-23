# --- CÓDIGO FINAL DO DETECTOR DE PADRÕES (v3.0 com busca "Head-First") ---

import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import yfinance as yf

print("Iniciando o processo de análise de padrões...")

# --- PASSO 1: AQUISIÇÃO E PREPARAÇÃO DOS DADOS ---

print("Buscando dados históricos do Bitcoin (BTC-USD) da internet...")
df_4h = yf.download(tickers='BTC-USD', period='2y',
                    interval='4h', auto_adjust=True, progress=False)

if df_4h.empty:
    print("Não foi possível buscar os dados. Verifique sua conexão com a internet ou o ticker 'BTC-USD'.")
    exit()

print("Dados de 4 horas carregados com sucesso.")

# --- PASSO 2: SUAVIZAÇÃO DE DADOS E DETECÇÃO DE EXTREMOS ---

# ==> OS PARÂMETROS PRINCIPAIS PARA AJUSTE <==
# Proeminência como porcentagem do preço médio para ser adaptável.
PROMINENCE_PERCENTUAL = 0.002  # Ex: 0.015 = 1.5%
# Duração máxima do padrão em número de velas.
MAX_VELAS_PADRAO = 150
# Regra de dominância da cabeça: ombro deve ter no máx 98% da altura da cabeça.
HEAD_SHOULDER_RATIO_THRESHOLD = 0.995
# Regra de simetria dos ombros: ombro menor deve ter no min 95% da altura do maior.
SYMMETRY_THRESHOLD = 0.9
# Janela da Média Móvel para suavização dos preços.
WINDOW_SMA = 3

# 1. CALCULAR A MÉDIA MÓVEL
print(f"Suavizando os dados com uma Média Móvel de {WINDOW_SMA} períodos...")
df_4h['SMA_Close'] = df_4h['Close'].rolling(window=WINDOW_SMA).mean()

# 2. REMOVER LINHAS VAZIAS
df_4h.dropna(inplace=True)

# 3. USAR A MÉDIA MÓVEL SUAVIZADA PARA A DETECÇÃO
preco_medio = float(df_4h['Close'].mean())
prominence_calculada = preco_medio * PROMINENCE_PERCENTUAL

precos = df_4h['SMA_Close'].values.reshape(-1)

indices_picos, _ = find_peaks(precos, prominence=prominence_calculada)
indices_vales, _ = find_peaks(-precos, prominence=prominence_calculada)

extremos = []
for i in indices_picos:
    extremos.append(
        {'data': df_4h.index[i], 'preco': precos[i], 'tipo': 'PICO'})
for i in indices_vales:
    extremos.append(
        {'data': df_4h.index[i], 'preco': precos[i], 'tipo': 'VALE'})

extremos.sort(key=lambda x: x['data'])
print(
    f"Com PROMINENCE calculada em ${prominence_calculada:.2f} ({PROMINENCE_PERCENTUAL * 100}%), foram encontrados {len(extremos)} extremos significativos.")

# --- PASSO 3: LÓGICA DE RECONHECIMENTO DE PADRÃO (ESTRATÉGIA "HEAD-FIRST") ---

lista_de_padroes = []
print(
    f"Iniciando a busca 'Head-First' por padrões (duração máx: {MAX_VELAS_PADRAO} velas)...")

# Iteramos pelos extremos, procurando por uma possível Cabeça.
for k in range(2, len(extremos) - 2):

    # --- BUSCA POR PADRÃO OCOI (Cabeça é um VALE) ---
    if extremos[k]['tipo'] == 'VALE':
        cabeca = extremos[k]
        pico1, ombro1, pico2, ombro2 = None, None, None, None

        # Busca para a ESQUERDA
        for j in range(k - 1, 0, -1):
            if extremos[j]['tipo'] == 'PICO':
                pico1 = extremos[j]
                for i in range(j - 1, -1, -1):
                    if extremos[i]['tipo'] == 'VALE':
                        ombro1 = extremos[i]
                        break
                break

        # Busca para a DIREITA
        if ombro1:
            for l in range(k + 1, len(extremos)):
                if extremos[l]['tipo'] == 'PICO':
                    pico2 = extremos[l]
                    for m in range(l + 1, len(extremos)):
                        if extremos[m]['tipo'] == 'VALE':
                            ombro2 = extremos[m]
                            break
                    break

        if all([ombro1, pico1, cabeca, pico2, ombro2]):
            idx_ombro1 = df_4h.index.get_loc(ombro1['data'])
            idx_ombro2 = df_4h.index.get_loc(ombro2['data'])
            if (idx_ombro2 - idx_ombro1) > MAX_VELAS_PADRAO:
                continue

            if cabeca['preco'] < ombro1['preco'] and cabeca['preco'] < ombro2['preco']:
                if (cabeca['preco'] / ombro1['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD) and \
                   (cabeca['preco'] / ombro2['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD):
                    if min(ombro1['preco'], ombro2['preco']) / max(ombro1['preco'], ombro2['preco']) > SYMMETRY_THRESHOLD:
                        padrao_encontrado_obj = {'tipo_padrao': 'OCOI', 'data_inicio': ombro1['data'], 'data_fim': ombro2['data'], 'preco_cabeca': cabeca[
                            'preco'], 'data_cabeca': cabeca['data'], 'preco_ombro1': ombro1['preco'], 'preco_ombro2': ombro2['preco']}
                        if padrao_encontrado_obj not in lista_de_padroes:
                            lista_de_padroes.append(padrao_encontrado_obj)

    # --- BUSCA POR PADRÃO OCO (Cabeça é um PICO) ---
    elif extremos[k]['tipo'] == 'PICO':
        cabeca = extremos[k]
        vale1, ombro1, vale2, ombro2 = None, None, None, None

        # Busca para a ESQUERDA
        for j in range(k - 1, 0, -1):
            if extremos[j]['tipo'] == 'VALE':
                vale1 = extremos[j]
                for i in range(j - 1, -1, -1):
                    if extremos[i]['tipo'] == 'PICO':
                        ombro1 = extremos[i]
                        break
                break

        # Busca para a DIREITA
        if ombro1:
            for l in range(k + 1, len(extremos)):
                if extremos[l]['tipo'] == 'VALE':
                    vale2 = extremos[l]
                    for m in range(l + 1, len(extremos)):
                        if extremos[m]['tipo'] == 'PICO':
                            ombro2 = extremos[m]
                            break
                    break

        if all([ombro1, vale1, cabeca, vale2, ombro2]):
            idx_ombro1 = df_4h.index.get_loc(ombro1['data'])
            idx_ombro2 = df_4h.index.get_loc(ombro2['data'])
            if (idx_ombro2 - idx_ombro1) > MAX_VELAS_PADRAO:
                continue

            if cabeca['preco'] > ombro1['preco'] and cabeca['preco'] > ombro2['preco']:
                if (ombro1['preco'] / cabeca['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD) and \
                   (ombro2['preco'] / cabeca['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD):
                    if min(ombro1['preco'], ombro2['preco']) / max(ombro1['preco'], ombro2['preco']) > SYMMETRY_THRESHOLD:
                        padrao_encontrado_obj = {'tipo_padrao': 'OCO', 'data_inicio': ombro1['data'], 'data_fim': ombro2['data'], 'preco_cabeca': cabeca[
                            'preco'], 'data_cabeca': cabeca['data'], 'preco_ombro1': ombro1['preco'], 'preco_ombro2': ombro2['preco']}
                        if padrao_encontrado_obj not in lista_de_padroes:
                            lista_de_padroes.append(padrao_encontrado_obj)

print(
    f"Busca finalizada. Total de {len(lista_de_padroes)} padrões encontrados.")

# --- PASSO 4: VERIFICAÇÃO DA CONFIRMAÇÃO E EXPORTAÇÃO ---
if lista_de_padroes:
    colunas_df = ['tipo_padrao', 'data_inicio', 'data_fim',
                  'preco_cabeca', 'data_cabeca', 'preco_ombro1', 'preco_ombro2']
    df_resultados = pd.DataFrame(lista_de_padroes, columns=colunas_df)

    # Ordena os resultados por data para que os mais recentes apareçam primeiro no CSV
    df_resultados.sort_values(by='data_inicio', ascending=False, inplace=True)

    nome_arquivo_saida = f'data/4h/{PROMINENCE_PERCENTUAL}_{MAX_VELAS_PADRAO}_{HEAD_SHOULDER_RATIO_THRESHOLD}_{SYMMETRY_THRESHOLD}_{WINDOW_SMA}.csv'
    df_resultados.to_csv(
        nome_arquivo_saida,
        index=False,
        float_format='%.2f',
        date_format='%Y-%m-%d %H:%M:%S'
    )
    print(
        f"\nResultados salvos com sucesso no arquivo '{nome_arquivo_saida}'!")
else:
    print("\nNenhum padrão foi encontrado com os parâmetros atuais.")
