import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Passo 1: Carregar o Dataset de Features ---
caminho_do_seu_arquivo = 'data/datasets/enriched/dataset_final_ml.csv'

try:
    df = pd.read_csv(caminho_do_seu_arquivo)
    print(
        f"Arquivo '{caminho_do_seu_arquivo}' carregado com sucesso. {df.shape[0]} linhas encontradas.")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{caminho_do_seu_arquivo}' não foi encontrado.")
    exit()

# --- Passo 2: Definir Features (X) e Rótulo (y) ---
### INÍCIO DA MODIFICAÇÃO ###
# Lista de features atualizada para incluir as novas colunas geradas.
# A coluna 'duracao_em_velas' foi removida pois não é gerada pelo script de enriquecimento.
colunas_features = [
    # Features Originais
    'altura_rel_cabeca', 'ratio_ombro_esquerdo', 'ratio_ombro_direito',
    'ratio_simetria_altura_ombros', 'neckline_slope', 'volume_breakout_ratio',
    'intervalo_em_minutos', 'duracao_em_velas',
    # Novas Features Adicionadas
    'dist_ombro1_cabeca', 'dist_cabeca_ombro2', 'ratio_simetria_temporal',
    'dif_altura_ombros_rel', 'extensao_ombro1', 'extensao_ombro2',
    'neckline_angle_rad', 'ruido_padrao'
]
coluna_label = 'label_humano'
### FIM DA MODIFICAÇÃO ###

# --- Passo 3: Limpeza e Preparação dos Dados ---
print("\n--- INICIANDO LIMPEZA E PREPARAÇÃO DOS DADOS ---")

# 3.1: Filtrar por labels válidos (0 ou 1)
print(
    f"Valores únicos na coluna '{coluna_label}' antes da limpeza: {df[coluna_label].unique()}")
df_limpo = df[df[coluna_label].isin([0, 1])].copy()
df_limpo[coluna_label] = df_limpo[coluna_label].astype(int)
print(
    f"Valores únicos na coluna '{coluna_label}' após filtrar por 0 e 1: {df_limpo[coluna_label].unique()}")
print(f"{df_limpo.shape[0]} linhas restantes após filtrar labels.")

### INÍCIO DA MODIFICAÇÃO ###
# 3.2: Remover linhas com valores NaN nas features ou no label
# Isso é crucial para que o treinamento não falhe.
colunas_necessarias = colunas_features + [coluna_label]
linhas_antes_nan = df_limpo.shape[0]

# Remove qualquer linha que tenha pelo menos um NaN nas colunas que usaremos
df_limpo.dropna(subset=colunas_necessarias, inplace=True)

linhas_depois_nan = df_limpo.shape[0]
linhas_removidas = linhas_antes_nan - linhas_depois_nan

print(f"\nRemovendo linhas com dados ausentes (NaN) nas features ou no label...")
print(f"{linhas_removidas} linhas foram removidas por conterem dados incompletos.")
print(
    f"{df_limpo.shape[0]} linhas válidas e completas restantes para o treinamento.")
### FIM DA MODIFICAÇÃO ###

if df_limpo.shape[0] < 20:  # Aumentei o limiar para garantir treino e teste
    print("ERRO: Poucos dados restantes após a limpeza. O modelo não pode ser treinado.")
    exit()

# A partir de agora, usamos o df_limpo
X = df_limpo[colunas_features]
y = df_limpo[coluna_label]

print(
    f"\nUsando {len(colunas_features)} features para prever '{coluna_label}'.")
print(f"Distribuição final das classes:")
print(f"  - Padrões Válidos (1):   {np.sum(y == 1)}")
print(f"  - Padrões Inválidos (0): {np.sum(y == 0)}")


# --- Passo 4: Dividir em Dados de Treino e Teste ---
print("\nDividindo o dataset em treino e teste (80/20)...")
# A função train_test_split embaralha os dados por padrão (shuffle=True).
# 'random_state' garante que o embaralhamento seja o mesmo a cada execução (reprodutibilidade).
# 'stratify=y' mantém a proporção de classes nos sets de treino e teste, importante para dados desbalanceados.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Tamanho do set de treino: {len(X_train)}")
print(f"Tamanho do set de teste: {len(X_test)}")


# --- Passo 5: Pré-processamento (Escalonamento das Features) ---
print("\nRealizando escalonamento das features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Passo 6: Construir e Treinar o Modelo (Random Forest) ---
print("\nConstruindo e treinando o modelo RandomForestClassifier...")
# n_estimators: número de árvores na floresta.
# class_weight='balanced': ajusta os pesos das classes para penalizar mais os erros na classe minoritária.
model = RandomForestClassifier(
    n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Treinamento concluído.")


# --- Passo 7: Avaliar o Modelo ---
print("\n--- AVALIAÇÃO FINAL DO MODELO ---")
y_pred = model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
print(f"Acurácia no Set de Teste: {accuracy:.2%}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=[
      'Inválido (0)', 'Válido (1)'], labels=[0, 1], zero_division=0))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Previsto: Inválido', 'Previsto: Válido'],
            yticklabels=['Real: Inválido', 'Real: Válido'])
plt.title('Matriz de Confusão')
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Previsto')
plt.show()

# --- NOVO: Análise da Importância das Features ---
print("\nImportância das Features (Feature Importance):")
feature_importances = pd.DataFrame(
    model.feature_importances_,
    index=colunas_features,
    columns=['importance']
).sort_values('importance', ascending=False)

print(feature_importances)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances.importance,
            y=feature_importances.index, palette='viridis')
plt.title('Importância de Cada Feature para o Modelo')
plt.xlabel('Importância')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
