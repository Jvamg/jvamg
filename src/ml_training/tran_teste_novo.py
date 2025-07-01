#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Final para Treinamento do Classificador de Qualidade de Padrões.

Este script treina o modelo LightGBM final com os hiperparâmetros otimizados
previamente encontrados, avalia sua performance no conjunto de teste e,
finalmente, gera um arquivo CSV com os erros do modelo para análise manual.
"""

# --- Importação das Bibliotecas Essenciais ---
import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Função principal que orquestra todo o pipeline de treinamento e análise."""

    # --- Passo 1: Carregar o Dataset de Features ---
    caminho_do_seu_arquivo = 'data/datasets/enriched/dataset_final_ml.csv'

    try:
        df = pd.read_csv(caminho_do_seu_arquivo)
        print(
            f"Arquivo '{caminho_do_seu_arquivo}' carregado com sucesso. {df.shape[0]} linhas encontradas.")
    except FileNotFoundError:
        print(
            f"ERRO: O arquivo '{caminho_do_seu_arquivo}' não foi encontrado.")
        return

    # --- Passo 2: Definir Features (X) e Rótulo (y) ---
    colunas_features = [
        'altura_rel_cabeca', 'ratio_ombro_esquerdo', 'ratio_ombro_direito',
        'ratio_simetria_altura_ombros', 'neckline_slope', 'volume_breakout_ratio',
        'intervalo_em_minutos', 'duracao_em_velas', 'dist_ombro1_cabeca',
        'dist_cabeca_ombro2', 'ratio_simetria_temporal', 'dif_altura_ombros_rel',
        'extensao_ombro1', 'extensao_ombro2', 'neckline_angle_rad', 'ruido_padrao'
    ]
    coluna_label = 'label_humano'

    # --- Passo 3: Limpeza e Preparação dos Dados ---
    print("\n--- INICIANDO LIMPEZA E PREPARAÇÃO DOS DADOS ---")

    df_limpo = df[df[coluna_label].isin([0, 1])].copy()
    df_limpo[coluna_label] = df_limpo[coluna_label].astype(int)

    colunas_necessarias = colunas_features + [coluna_label]
    df_limpo.dropna(subset=colunas_necessarias, inplace=True)

    print(
        f"{df_limpo.shape[0]} linhas válidas e completas restantes para o treinamento.")
    if df_limpo.shape[0] < 50:
        print("ERRO: Poucos dados restantes após a limpeza.")
        return

    X = df_limpo[colunas_features]
    y = df_limpo[coluna_label]

    print("\nDistribuição final das classes:")
    print(f"  - Padrões Válidos (1):   {np.sum(y == 1)}")
    print(f"  - Padrões Inválidos (0): {np.sum(y == 0)}")

    # --- Passo 4: Dividir em Dados de Treino e Teste ---
    print("\nDividindo o dataset em treino e teste (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- Passo 5: Pré-processamento (Escalonamento) ---
    print("\nRealizando escalonamento das features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # --- Passo 6: Treinar o Modelo FINAL com os Melhores Parâmetros ---
    print("\n--- Treinando o modelo final com os melhores parâmetros encontrados ---")

    # Estes são os melhores parâmetros que o RandomizedSearchCV encontrou na execução anterior.
    # Colamos eles aqui para treinar o modelo definitivo.
    melhores_parametros = {
        'learning_rate': 0.09550820367170992,
        'max_depth': 5,
        'n_estimators': 479,
        'num_leaves': 32,
        'reg_alpha': 0.2017192023353962,
        'reg_lambda': 0.8957635956735194,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    # Instanciando o modelo com esses parâmetros
    model = lightgbm.LGBMClassifier(**melhores_parametros)

    # Treinando o modelo com todos os dados de treino
    model.fit(X_train_scaled, y_train)
    print("Treinamento do modelo otimizado concluído.")

    # --- Passo 7: Avaliar o Modelo OTIMIZADO ---
    print("\n--- AVALIAÇÃO FINAL DO MODELO OTIMIZADO ---")
    y_pred = model.predict(X_test_scaled)

    print(f"\nAcurácia no Set de Teste: {accuracy_score(y_test, y_pred):.2%}")

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred,
          target_names=['Inválido (0)', 'Válido (1)']))

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Previsto: Inválido', 'Previsto: Válido'],
                yticklabels=['Real: Inválido', 'Real: Válido'])
    plt.title('Matriz de Confusão - Modelo LightGBM Otimizado')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.show()

    print("\nImportância das Features (Feature Importance):")
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importances)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importances, x='importance',
                y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Importância de Cada Feature para o Modelo LightGBM Otimizado')
    plt.xlabel('Importância')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

    # Substitua o Passo 8 inteiro por este:

    # --- Passo 8: Análise de Erros (Versão Corrigida) ---
    print("\n--- INICIANDO ANÁLISE DE ERROS ---")

    # Juntando os dados de teste com as previsões para facilitar a análise
    # REMOVEMOS o .reset_index() para manter os índices originais do df_limpo
    df_teste_analise = X_test.copy()
    df_teste_analise['label_real'] = y_test
    df_teste_analise['previsao_modelo'] = y_pred

    # Filtrando apenas as linhas onde o modelo errou
    erros_df = df_teste_analise[df_teste_analise['label_real']
                                != df_teste_analise['previsao_modelo']]

    print(
        f"\nO modelo errou em {len(erros_df)} dos {len(df_teste_analise)} exemplos de teste.")

    falsos_positivos = erros_df[erros_df['label_real'] == 0]
    falsos_negativos = erros_df[erros_df['label_real'] == 1]

    print(
        f"Total de Falsos Positivos (ruído que o modelo aceitou): {len(falsos_positivos)}")
    print(
        f"Total de Falsos Negativos (padrões bons que o modelo rejeitou): {len(falsos_negativos)}")

    # Agora, esta linha vai funcionar, pois os índices em erros_df correspondem aos de df_limpo
    # Usamos o índice de erros_df para buscar as linhas completas (com ticker, datas, etc.)
    erros_completos_df = df_limpo.loc[erros_df.index]
    # Adicionamos a previsão ao lado
    erros_completos_df['previsao_modelo'] = erros_df['previsao_modelo']

    try:
        erros_completos_df.to_csv('analise_de_erros.csv', index=False)
        print("\nArquivo 'analise_de_erros.csv' salvo com sucesso!")
        print("Sua tarefa agora é abrir este arquivo e começar o trabalho de detetive: olhe os gráficos correspondentes a cada erro.")
    except Exception as e:
        print(f"\nNão foi possível salvar o arquivo de erros: {e}")


if __name__ == '__main__':
    main()
