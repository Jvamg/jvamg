import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_model(dataset_path: str = 'data/datasets/filtered/dataset_labeled.csv', model_output_path: str = 'data/models/feature_model.joblib'):
    """
    Carrega o dataset rotulado, treina um modelo RandomForestClassifier
    e salva o modelo treinado.
    """
    print("--- Início do Pipeline de Treinamento de Modelo ---")

    # --- 1. Carregamento de Dados ---
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset '{dataset_path}' carregado com sucesso. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo do dataset não encontrado em '{dataset_path}'.")
        print("Certifique-se de que o caminho está correto e o arquivo de labeling foi gerado.")
        return

    # --- 2. Preparação de Features (X) e Alvo (y) ---
    print("\n--- 2. Preparação dos Dados ---")
    
    # Remove padrões que não foram rotulados ou que foram marcados como erro (-1)
    df_clean = df.dropna(subset=['label_humano'])
    df_clean = df_clean[df_clean['label_humano'] != -1]
    
    if df_clean.empty:
        print("❌ ERRO: Não há dados rotulados válidos para treinar o modelo.")
        return
        
    print(f"Total de exemplos válidos para treinamento: {len(df_clean)}")
    print(f"Distribuição dos labels:\n{df_clean['label_humano'].value_counts(normalize=True)}")

    # Define as 'pistas' (features) que o modelo usará para aprender
    features = [
    # Features de Escala que já tínhamos
    'duracao_em_velas',
    'intervalo_em_minutos',

    # Novas Features Geométricas (o "ouro")
    'altura_rel_cabeca',
    'ratio_ombro_esquerdo',
    'ratio_ombro_direito',
    'ratio_simetria_altura_ombros',
    'neckline_slope'
    ]
    
    print(f"Exemplos antes da limpeza de features: {len(df_clean)}")
    df_clean.dropna(subset=features, inplace=True)
    print(f"Exemplos após limpeza completa: {len(df_clean)}")

    X = df_clean[features]
    y = df_clean['label_humano'].astype(int) # O alvo (target) que queremos prever

    print(f"Features selecionadas para X: {X.columns.tolist()}")
    print(f"Alvo selecionado para y: 'label_humano'")

    # --- 3. Divisão de Dados (Treino e Teste) ---
    print("\n--- 3. Divisão em Conjuntos de Treino e Teste ---")
    # Usamos stratify=y para garantir que a proporção de 0s e 1s seja a mesma
    # nos dados de treino e de teste, tornando a avaliação mais justa.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,    # 20% dos dados para teste
        random_state=42,  # Para reprodutibilidade dos resultados
        stratify=y
    )
    print(f"Tamanho do conjunto de treino: {len(X_train)} amostras")
    print(f"Tamanho do conjunto de teste: {len(X_test)} amostras")

    # --- 4. Seleção e Treinamento do Modelo ---
    print("\n--- 4. Treinamento do Modelo RandomForest ---")
    # RandomForest é um conjunto de "árvores de decisão", ótimo para dados tabulares.
    # n_estimators é o número de árvores na floresta.
    model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    
    print("Treinando o modelo com os dados de treino...")
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    print(f"Acurácia OOB (Out-of-Bag): {model.oob_score_:.4f}")


    # --- 5. Avaliação de Performance ---
    print("\n--- 5. Avaliação de Performance no Conjunto de Teste ---")
    y_pred = model.predict(X_test)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia Geral: {accuracy:.4f}")

    # Matriz de Confusão
    print("\nMatriz de Confusão:")
    # Formato:
    # [[Verdadeiro Negativo (TN), Falso Positivo (FP)],
    #  [Falso Negativo (FN),    Verdadeiro Positivo (TP)]]
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Relatório de Classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Ruim (0)', 'Bom (1)']))

    # --- 6. Importância das Features ---
    print("\n--- 6. Análise de Importância das Features ---")
    # Mostra o quão importante cada feature foi para a decisão do modelo
    feature_importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importances)


    # --- 7. Salvamento do Modelo ---
    print(f"\n--- 7. Salvando o modelo treinado em '{model_output_path}' ---")
    joblib.dump(model, model_output_path)
    print("Modelo salvo com sucesso!")
    print("\n--- Pipeline de Treinamento Finalizado ---")

if __name__ == '__main__':
    # Você pode alterar o caminho do dataset aqui se necessário
    caminho_dataset = 'data/datasets/enriched/dataset_final_ml.csv'
    train_model(dataset_path=caminho_dataset)