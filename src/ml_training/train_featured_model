import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Passo 1: Carregar o Dataset de Features ---
# Usei o caminho do seu log de erro
caminho_do_seu_arquivo = 'data/datasets/enriched/dataset_final_ml.csv'

try:
    df = pd.read_csv(caminho_do_seu_arquivo)
    print(
        f"Arquivo '{caminho_do_seu_arquivo}' carregado com sucesso. {df.shape[0]} linhas encontradas.")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{caminho_do_seu_arquivo}' não foi encontrado.")
    exit()

# --- Passo 2: Definir Features (X) e Rótulo (y) ---
colunas_features = [
    'altura_rel_cabeca', 'ratio_ombro_esquerdo', 'ratio_ombro_direito',
    'ratio_simetria_altura_ombros', 'neckline_slope', 'volume_breakout_ratio',
    'duracao_em_velas'
]
coluna_label = 'label_humano'

# --- NOVO! Passo 2.5: Limpeza e Verificação dos Dados ---
print("\nRealizando limpeza e verificação dos dados...")
print(
    f"Valores únicos na coluna '{coluna_label}' antes da limpeza: {df[coluna_label].unique()}")

# Mantém apenas as linhas onde o label é 0 ou 1, descartando quaisquer outros.
df_filtrado = df[df[coluna_label].isin([0, 1])].copy()

# Converte o tipo da coluna para inteiro, para garantir que não há floats (ex: 1.0)
df_filtrado[coluna_label] = df_filtrado[coluna_label].astype(int)

print(
    f"Valores únicos na coluna '{coluna_label}' após a limpeza: {df_filtrado[coluna_label].unique()}")
print(
    f"{df_filtrado.shape[0]} padrões restantes após filtrar por labels 0 e 1.")

if df_filtrado.shape[0] < 10:
    print("ERRO: Poucos dados restantes após a limpeza. Verifique a coluna 'label_humano' no seu CSV.")
    exit()

# A partir de agora, usamos o df_filtrado
X = df_filtrado[colunas_features]
y = df_filtrado[coluna_label]

print(
    f"\nUsando {len(colunas_features)} features para prever '{coluna_label}'.")
print(f"Total de padrões válidos (1): {np.sum(y == 1)}")
print(f"Total de padrões inválidos (0): {np.sum(y == 0)}")


# --- Passo 3: Dividir em Dados de Treino e Teste ---
print("\nDividindo o dataset em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Tamanho do set de treino: {len(X_train)}")
print(f"Tamanho do set de teste: {len(X_test)}")


# --- Passo 4: Pré-processamento (Escalonamento das Features) ---
print("\nRealizando escalonamento das features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Passo 5: Construir e Treinar o Modelo (Random Forest) ---
print("\nConstruindo e treinando o modelo RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Treinamento concluído.")


# --- Passo 6: Avaliar o Modelo ---
print("\n--- AVALIAÇÃO FINAL DO MODELO ---")
y_pred = model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
print(f"Acurácia no Set de Teste: {accuracy:.2%}")

print("\nRelatório de Classificação:")
# Adicionado o parâmetro 'labels' para tornar a função mais robusta
print(classification_report(y_test, y_pred, target_names=[
      'Inválido (0)', 'Válido (1)'], labels=[0, 1]))

print("Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Inválido (Previsto)', 'Válido (Previsto)'],
            yticklabels=['Inválido (Real)', 'Válido (Real)'])
plt.title('Matriz de Confusão - Modelo de Features')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.show()
