import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# --- PARÂMETROS GERAIS ---
ARQUIVO_ETIQUETAS_REAIS = 'dataset_verificado.csv'
TAMANHO_JANELA_FIXO = 100

# --- FUNÇÕES DE GERAÇÃO SINTÉTICA ---


def gerar_no_pattern(pontos=100):
    return np.random.randn(pontos).cumsum() / 10 + 1.5


def gerar_bull_flag(pontos=100):
    mastro = np.linspace(1.0, 2.0, pontos//2)
    bandeira = np.linspace(2.0, 1.8, pontos//2)
    return np.concatenate([mastro, bandeira]) + np.random.normal(0, 0.05, pontos)


# --- ESTÁGIO 1: MONTAGEM DO DATASET BINÁRIO ---
X = []
y = []
mapa_etiquetas_str_para_int = {'OCO': 1, 'OCOI': 1}

print(
    f"Carregando exemplos REAIS de OCO/OCOI do arquivo '{ARQUIVO_ETIQUETAS_REAIS}'...")
try:
    df_etiquetas_reais = pd.read_csv(ARQUIVO_ETIQUETAS_REAIS, parse_dates=[
                                     'data_inicio', 'data_fim'])
except Exception as e:
    print(f"Erro ao ler o arquivo CSV: {e}")
    exit()

print("\nProcessando exemplos POSITIVOS (Reais)...")
for index, linha in df_etiquetas_reais.iterrows():
    if linha['tipo_padrao'] in ['OCO', 'OCOI']:
        try:
            dados_janela = yf.download(
                tickers='BTC-USD', start=linha['data_inicio'], end=linha['data_fim'], interval='4h', progress=False)
            if not dados_janela.empty:
                precos = dados_janela['Close'].values
                janela_reamostrada = resample(precos, TAMANHO_JANELA_FIXO)
                min_val, max_val = np.min(
                    janela_reamostrada), np.max(janela_reamostrada)
                janela_normalizada = (janela_reamostrada - min_val) / (max_val - min_val) if (
                    max_val - min_val) > 0 else np.zeros(TAMANHO_JANELA_FIXO)

                # --- A CORREÇÃO FINAL ESTÁ AQUI ---
                # Usamos .flatten() para garantir que o array seja sempre 1D (formato (100,))
                X.append(janela_normalizada.flatten())
                y.append(1)
        except Exception as e:
            print(f"Erro ao processar linha {index} dos dados reais: {e}")

num_exemplos_reais_validos = sum(1 for label in y if label == 1)
print(
    f"Total de {num_exemplos_reais_validos} exemplos positivos válidos foram carregados.")

print("\nProcessando exemplos NEGATIVOS (Sintéticos)...")
for i in range(num_exemplos_reais_validos):
    if np.random.rand() > 0.5:
        padrao_gerado = gerar_no_pattern()
    else:
        padrao_gerado = gerar_bull_flag()
    X.append(padrao_gerado)  # Funções sintéticas já retornam 1D (100,)
    y.append(0)

# --- O RESTANTE DO CÓDIGO PERMANECE O MESMO ---
if not X:
    print("\nNenhum dado de treinamento foi gerado.")
    exit()

print("\nPreparando dataset final para a IA...")
# Agora esta linha vai funcionar, pois todos os itens em X são uniformes
X = np.array(X)
X = X.reshape(-1, TAMANHO_JANELA_FIXO, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(
    f"Dataset pronto: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste.")

print("\nConstruindo o modelo LSTM para classificação binária...")
model = Sequential([
    LSTM(units=50, input_shape=(TAMANHO_JANELA_FIXO, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
checkpoint_callback = ModelCheckpoint(
    filepath="modelo_especialista_oco.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
model.summary()

print("\nIniciando o treinamento do modelo especialista em OCO...")
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback],
    verbose=1
)

print("\nModelo especialista treinado. Avaliando...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nAcurácia final no conjunto de teste: {accuracy * 100:.2f}%')
print("✅ Modelo especialista salvo como 'modelo_especialista_oco.keras'.")
