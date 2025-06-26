import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# --- PARÂMETROS GERAIS ---
ARQUIVO_ETIQUETAS_REAIS = 'dataset_verificado.csv'
TAMANHO_JANELA_FIXO = 100
NUM_CLASSES = 5  # OCO/OCOI, Topo Duplo, Sem Padrão, Bandeira de Alta, Bandeira de Baixa
mapa_etiquetas_str_para_int = {
    'OCO': 0, 'Ombro-Cabeça-Ombro': 0, 'OCOI': 0,
    'TopoDuplo': 1, 'Topo Duplo': 1,
    'Sem Padrão': 2,
    'Bandeira de Alta': 3,
    'Bandeira de Baixa': 4
}

# ======================================================================
# === FASE 1: PRÉ-TREINAMENTO COM DADOS SINTÉTICOS ("O SIMULADOR") ===
# ======================================================================

# --- Funções de Geração de Dados Sintéticos (como antes) ---


def gerar_head_and_shoulders(pontos=100):
    p1 = np.linspace(1, 1.5, pontos//5)
    p2 = np.linspace(1.5, 1.2, pontos//5)
    p3 = np.linspace(1.2, 1.8, pontos//5)
    p4 = np.linspace(1.8, 1.3, pontos//5)
    p5 = np.linspace(1.3, 1.6, pontos//5)
    return np.concatenate([p1, p2, p3, p4, p5]) + np.random.normal(0, 0.05, pontos)


def gerar_double_top(pontos=100):
    p1 = np.linspace(1, 1.6, pontos//5)
    p2 = np.linspace(1.6, 1.2, pontos//5)
    p3 = np.linspace(1.2, 1.6, pontos//5)
    p4 = np.linspace(1.6, 1.0, pontos//5)
    p5 = np.linspace(1.0, 1.3, pontos//5)
    return np.concatenate([p1, p2, p3, p4, p5]) + np.random.normal(0, 0.05, pontos)


def gerar_no_pattern(pontos=100):
    return np.random.randn(pontos).cumsum()/10+1.5


def gerar_bull_flag(pontos=100):
    mastro = np.linspace(1.0, 2.0, pontos//2)
    bandeira = np.linspace(2.0, 1.8, pontos//2)
    return np.concatenate([mastro, bandeira]) + np.random.normal(0, 0.05, pontos)


def gerar_bear_flag(pontos=100):
    mastro = np.linspace(2.0, 1.0, pontos//2)
    bandeira = np.linspace(1.0, 1.2, pontos//2)
    return np.concatenate([mastro, bandeira]) + np.random.normal(0, 0.05, pontos)

def normalize_window(window_data):
    """Normaliza uma janela de dados para o intervalo [0, 1] (Min-Max scaling)."""
    min_val, max_val = np.min(window_data), np.max(window_data)
    if (max_val - min_val) > 0:
        return (window_data - min_val) / (max_val - min_val)
    return np.zeros_like(window_data)


# --- Montagem do Dataset Sintético ---
X_sintetico, y_sintetico = [], []
num_amostras_sinteticas = 1000  # Geramos bastante dados sintéticos
print(
    f"FASE 1: Gerando {num_amostras_sinteticas * NUM_CLASSES} amostras de dados sintéticos...")
for _ in range(num_amostras_sinteticas):
    X_sintetico.append(normalize_window(gerar_head_and_shoulders()))
    y_sintetico.append(0)
    X_sintetico.append(normalize_window(gerar_double_top()))
    y_sintetico.append(1)
    X_sintetico.append(normalize_window(gerar_no_pattern()))
    y_sintetico.append(2)
    X_sintetico.append(normalize_window(gerar_bull_flag()))
    y_sintetico.append(3)
    X_sintetico.append(normalize_window(gerar_bear_flag()))
    y_sintetico.append(4)

X_sintetico = np.array(X_sintetico).reshape(-1, TAMANHO_JANELA_FIXO, 1)
y_sintetico = to_categorical(y_sintetico, num_classes=NUM_CLASSES)

# --- Construção do Modelo ---
print("\nConstruindo a arquitetura do modelo LSTM...")
model = Sequential([
    LSTM(units=50, input_shape=(TAMANHO_JANELA_FIXO, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- Treinamento Base ---
print("\nIniciando pré-treinamento com dados sintéticos...")
model.fit(X_sintetico, y_sintetico, epochs=15,
          batch_size=32, validation_split=0.1, verbose=1)
print("✅ Pré-treinamento concluído. O modelo agora tem um conhecimento base.")


# ======================================================================
# === FASE 2: AJUSTE FINO COM DADOS REAIS ("AS HORAS DE VOO REAIS") ===
# ======================================================================

# --- Carregar e Processar os Dados Reais ---
print(
    f"\nFASE 2: Carregando etiquetas do arquivo real '{ARQUIVO_ETIQUETAS_REAIS}' para o ajuste fino...")
try:
    df_etiquetas_reais = pd.read_csv(ARQUIVO_ETIQUETAS_REAIS, parse_dates=[
                                     'data_inicio', 'data_fim'])
except Exception as e:
    print(f"Não foi possível ler o arquivo de dados reais: {e}")
    exit()

X_real, y_real = [], []
print("Processando dados reais...")
for index, linha in df_etiquetas_reais.iterrows():
    try:
        dados_janela = yf.download(
            tickers='BTC-USD', start=linha['data_inicio'], end=linha['data_fim'], interval='4h')
        if not dados_janela.empty:
            precos = dados_janela['Close'].values
            janela_reamostrada = resample(precos, TAMANHO_JANELA_FIXO)
            janela_normalizada = normalize_window(janela_reamostrada)
            X_real.append(janela_normalizada)
            y_real.append(mapa_etiquetas_str_para_int[linha['tipo_padrao']])
    except Exception as e:
        print(f"Erro ao processar linha {index} dos dados reais: {e}")

if not X_real:
    print("\nNenhum dado real foi processado para o ajuste fino. Salvando o modelo pré-treinado.")
    model.save('modelo_apenas_sintetico.keras')
    exit()

X_real = np.array(X_real).reshape(-1, TAMANHO_JANELA_FIXO, 1)
y_real = to_categorical(y_real, num_classes=NUM_CLASSES)
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42)

# --- A Mágica do Fine-Tuning ---
# Re-compilamos o modelo com uma TAXA DE APRENDIZAGEM MUITO BAIXA
taxa_aprendizagem_baixa = 0.0001
print(
    f"\nRe-compilando o modelo para o ajuste fino com learning rate de {taxa_aprendizagem_baixa}...")
model.compile(
    optimizer=Adam(learning_rate=taxa_aprendizagem_baixa),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Configuramos o Checkpoint para salvar o MELHOR modelo durante o ajuste fino
checkpoint_callback_ft = ModelCheckpoint(
    filepath="data\models\melhor_modelo.keras",  # Nome do arquivo final
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

print("\nIniciando o ajuste fino com dados reais...")
model.fit(
    X_train_real, y_train_real,
    epochs=40,  # Mais épocas podem ser úteis no ajuste fino com poucos dados
    batch_size=8,  # Lotes menores também podem ajudar
    validation_data=(X_test_real, y_test_real),
    callbacks=[checkpoint_callback_ft],
    verbose=1
)

print("\n✅ Treinamento e ajuste fino concluídos! O modelo especialista está salvo como 'modelo_final_ajustado.keras'.")
