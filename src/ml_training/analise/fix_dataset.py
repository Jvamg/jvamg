# fix_dataset.py (versão corrigida)

import pandas as pd
import numpy as np


def fix_and_clean_dataset(input_path: str, output_path: str):
    """
    Carrega o dataset rotulado, recalcula a duração dos padrões com base
    nas datas ajustadas, remove colunas obsoletas e salva um novo
    dataset limpo e consistente.

    Args:
        input_path (str): Caminho para o arquivo dataset_labeled.csv.
        output_path (str): Caminho onde o novo arquivo corrigido será salvo.
    """
    print("--- Iniciando Script de Correção e Sincronização de Dataset ---")

    # --- 1. Carregamento de Dados ---
    try:
        df = pd.read_csv(input_path)
        print(
            f"Dataset '{input_path}' carregado com sucesso. Total de {len(df)} registros.")
    except FileNotFoundError:
        print(f"🚨 ERRO: Arquivo do dataset não encontrado em '{input_path}'.")
        print("Verifique se o caminho está correto e se o arquivo de labeling existe.")
        return

    # --- 2. Processamento e Correção ---
    print("\nIniciando processamento e correção...")

    # Garante que as colunas de data sejam do tipo datetime
    df['data_inicio'] = pd.to_datetime(df['data_inicio'])
    df['data_fim'] = pd.to_datetime(df['data_fim'])
    print(" -> Colunas de data convertidas para o formato datetime.")

    # --- MUDANÇA: CORREÇÃO DA CONVERSÃO DE TIMEDELTA ---
    # 1. Primeiro, substituímos a abreviação não padrão 'wk' pela padrão 'w' que o pandas entende.
    #    O case=False garante que funcione para 'wk' ou 'WK'. regex=False é por segurança.
    intervalos_corrigidos = df['intervalo'].str.replace(
        'wk', 'w', case=False, regex=False)
    print(" -> Strings de intervalo normalizadas (ex: '1wk' -> '1w').")

    # 2. Agora, com os intervalos corrigidos, a conversão para Timedelta funcionará.
    timedelta_intervalo = pd.to_timedelta(intervalos_corrigidos)

    # 3. Recalcula a 'duracao_em_velas' de forma vetorizada
    duracao_total = df['data_fim'] - df['data_inicio']
    df['duracao_em_velas'] = (
        duracao_total / timedelta_intervalo).round().astype(int)
    print(" -> Coluna 'duracao_em_velas' recalculada e sincronizada com as datas ajustadas.")
    # --- FIM DA MUDANÇA ---

    # Remove as colunas obsoletas e inconsistentes
    colunas_para_remover = ['idx_inicio', 'idx_fim']
    df.drop(columns=colunas_para_remover, inplace=True, errors='ignore')
    print(
        f" -> Colunas obsoletas ({', '.join(colunas_para_remover)}) removidas.")

    # --- 3. Saída ---
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Sucesso! Dataset corrigido e salvo em '{output_path}'.")
        print(
            f"O novo dataset contém {df.shape[0]} linhas e {df.shape[1]} colunas.")
        print("\nPrévia das colunas 'duracao_em_velas' e 'intervalo':")
        print(df[['intervalo', 'duracao_em_velas']].head())
    except Exception as e:
        print(f"🚨 ERRO: Não foi possível salvar o arquivo de saída. Erro: {e}")


if __name__ == '__main__':
    # Defina os caminhos para os seus arquivos
    ARQUIVO_ROTULADO = 'data/datasets/filtered/dataset_labeled.csv'
    ARQUIVO_FINAL_CORRIGIDO = 'data/datasets/filtered/dataset_corrected_final.csv'

    fix_and_clean_dataset(ARQUIVO_ROTULADO, ARQUIVO_FINAL_CORRIGIDO)
