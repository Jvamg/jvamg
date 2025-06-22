# --- CÓDIGO FINAL COM TODAS AS REGRAS E LÓGICA COMPLETA E CORRIGIDA ---

import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import yfinance as yf


def encontrar_padroes_em_historico(ticker='BTC-USD', periodo='2y', intervalo='4h'):
    """
    Busca dados históricos de um ticker e aplica regras para encontrar
    padrões OCO e OCOI, salvando o resultado em um CSV.
    """
    print(f"Iniciando busca de padrões para {ticker}...")

    # --- PASSO 1: AQUISIÇÃO E PREPARAÇÃO DOS DADOS ---

    # A MUDANÇA É AQUI: Usando as variáveis 'periodo' e 'intervalo' da sua função
    df = yf.download(tickers=ticker, period=periodo, interval=intervalo)

    if df.empty:
        print(f"Não foi possível buscar dados para {ticker}.")
        return None

    print(f"Dados de {intervalo} carregados com sucesso para {ticker}.")

    # --- PASSO 2: DETECÇÃO DE PICOS E VALES (EXTREMOS) ---
    PROMINENCE_VALUE = 1000
    MAX_VELAS_PADRAO = 120
    HEAD_SHOULDER_RATIO_THRESHOLD = 0.98

    precos = df['Close'].values
    indices_picos, _ = find_peaks(precos, prominence=PROMINENCE_VALUE)
    indices_vales, _ = find_peaks(-precos, prominence=PROMINENCE_VALUE)

    extremos = []
    for i in indices_picos:
        extremos.append(
            {'data': df.index[i], 'preco': precos[i], 'tipo': 'PICO'})
    for i in indices_vales:
        extremos.append(
            {'data': df.index[i], 'preco': precos[i], 'tipo': 'VALE'})

    extremos.sort(key=lambda x: x['data'])
    print(
        f"Com PROMINENCE = {PROMINENCE_VALUE}, foram encontrados {len(extremos)} extremos significativos.")

    # --- PASSO 3: LÓGICA DE RECONHECIMENTO DE PADRÃO ---
    lista_de_padroes = []
    print(
        f"Iniciando a busca avançada por padrões (duração máx: {MAX_VELAS_PADRAO} velas)...")

    for i in range(len(extremos)):
        ponto_inicial = extremos[i]

        if ponto_inicial['tipo'] == 'PICO':
            ombro1 = ponto_inicial
            for j in range(i + 1, len(extremos)):
                if extremos[j]['tipo'] == 'VALE':
                    idx_ombro1 = df.index.get_loc(ombro1['data'])
                    idx_atual = df.index.get_loc(extremos[j]['data'])
                    if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                        break
                    for k in range(j + 1, len(extremos)):
                        if extremos[k]['tipo'] == 'PICO':
                            idx_atual = df.index.get_loc(extremos[k]['data'])
                            if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                                break
                            cabeca = extremos[k]
                            for l in range(k + 1, len(extremos)):
                                if extremos[l]['tipo'] == 'VALE':
                                    idx_atual = df.index.get_loc(
                                        extremos[l]['data'])
                                    if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                                        break
                                    for m in range(l + 1, len(extremos)):
                                        if extremos[m]['tipo'] == 'PICO':
                                            idx_atual = df.index.get_loc(
                                                extremos[m]['data'])
                                            if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                                                break
                                            ombro2 = extremos[m]
                                            if cabeca['preco'] > ombro1['preco'] and cabeca['preco'] > ombro2['preco']:
                                                if (ombro1['preco'] / cabeca['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD) and (ombro2['preco'] / cabeca['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD):
                                                    if min(ombro1['preco'], ombro2['preco']) / max(ombro1['preco'], ombro2['preco']) > 0.7:
                                                        padrao_encontrado = {
                                                            'tipo_padrao': 'OCO', 'data_inicio': ombro1['data'], 'data_fim': ombro2['data'], 'preco_cabeca': cabeca['preco'], 'data_cabeca': cabeca['data']}
                                                        if padrao_encontrado not in lista_de_padroes:
                                                            lista_de_padroes.append(
                                                                padrao_encontrado)
                                            break
                                    break
                                break
                            break
                        break
        elif ponto_inicial['tipo'] == 'VALE':
            ombro1 = ponto_inicial
            for j in range(i + 1, len(extremos)):
                if extremos[j]['tipo'] == 'PICO':
                    idx_ombro1 = df.index.get_loc(ombro1['data'])
                    idx_atual = df.index.get_loc(extremos[j]['data'])
                    if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                        break
                    for k in range(j + 1, len(extremos)):
                        if extremos[k]['tipo'] == 'VALE':
                            idx_atual = df.index.get_loc(extremos[k]['data'])
                            if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                                break
                            cabeca = extremos[k]
                            for l in range(k + 1, len(extremos)):
                                if extremos[l]['tipo'] == 'PICO':
                                    idx_atual = df.index.get_loc(
                                        extremos[l]['data'])
                                    if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                                        break
                                    for m in range(l + 1, len(extremos)):
                                        if extremos[m]['tipo'] == 'VALE':
                                            idx_atual = df.index.get_loc(
                                                extremos[m]['data'])
                                            if (idx_atual - idx_ombro1) > MAX_VELAS_PADRAO:
                                                break
                                            ombro2 = extremos[m]
                                            if cabeca['preco'] < ombro1['preco'] and cabeca['preco'] < ombro2['preco']:
                                                if (cabeca['preco'] / ombro1['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD) and (cabeca['preco'] / ombro2['preco'] < HEAD_SHOULDER_RATIO_THRESHOLD):
                                                    if min(ombro1['preco'], ombro2['preco']) / max(ombro1['preco'], ombro2['preco']) > 0.7:
                                                        padrao_encontrado = {
                                                            'tipo_padrao': 'OCOI', 'data_inicio': ombro1['data'], 'data_fim': ombro2['data'], 'preco_cabeca': cabeca['preco'], 'data_cabeca': cabeca['data']}
                                                        if padrao_encontrado not in lista_de_padroes:
                                                            lista_de_padroes.append(
                                                                padrao_encontrado)
                                            break
                                    break
                                break
                            break
                        break

    print(
        f"Busca finalizada. Total de {len(lista_de_padroes)} padrões encontrados.")

    if lista_de_padroes:
        df_resultados = pd.DataFrame(lista_de_padroes)
        nome_arquivo_saida = f'candidatos_{ticker.replace("/", "_")}.csv'
        df_resultados.to_csv(nome_arquivo_saida, index=False)
        print(f"Resultados salvos com sucesso em '{nome_arquivo_saida}'!")
        return nome_arquivo_saida
    else:
        print("Nenhum padrão foi encontrado com os parâmetros atuais.")
        return None


if __name__ == '__main__':
    encontrar_padroes_em_historico(ticker='BTC-USD')
    print("\n" + "="*50 + "\n")
    encontrar_padroes_em_historico(ticker='ETH-USD', periodo='1y')
