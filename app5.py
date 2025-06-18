import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- ESTÁGIO 1: GERAÇÃO DE DADOS SINTÉTICOS ---

def gerar_head_and_shoulders(pontos=100):
    p1 = np.linspace(1, 1.5, pontos // 5)
    p2 = np.linspace(1.5, 1.2, pontos // 5)
    p3 = np.linspace(1.2, 1.8, pontos // 5)
    p4 = np.linspace(1.8, 1.3, pontos // 5)
    p5 = np.linspace(1.3, 1.6, pontos // 5)
    padrao = np.concatenate([p1, p2, p3, p4, p5])
    ruido = np.random.normal(0, 0.05, pontos)
    return padrao + ruido

def gerar_double_top(pontos=100):
    p1 = np.linspace(1, 1.6, pontos // 5)
    p2 = np.linspace(1.6, 1.2, pontos // 5)
    p3 = np.linspace(1.2, 1.6, pontos // 5)
    p4 = np.linspace(1.6, 1.0, pontos // 5)
    p5 = np.linspace(1.0, 1.3, pontos // 5)
    padrao = np.concatenate([p1, p2, p3, p4, p5])
    ruido = np.random.normal(0, 0.05, pontos)
    return padrao + ruido

def gerar_no_pattern(pontos=100):
    return np.random.randn(pontos).cumsum() / 10 + 1.5

# --- NOVAS FUNÇÕES PARA GERAR BANDEIRAS ---
def gerar_bull_flag(pontos=100):
    # Bandeira de Alta: Mastro de subida, seguido de uma pequena consolidação para baixo
    mastro = np.linspace(1.0, 2.0, pontos // 2)
    bandeira = np.linspace(2.0, 1.8, pontos // 2)
    padrao = np.concatenate([mastro, bandeira])
    ruido = np.random.normal(0, 0.05, pontos)
    return padrao + ruido

def gerar_bear_flag(pontos=100):
    # Bandeira de Baixa: Mastro de descida, seguido de uma pequena consolidação para cima
    mastro = np.linspace(2.0, 1.0, pontos // 2)
    bandeira = np.linspace(1.0, 1.2, pontos // 2)
    padrao = np.concatenate([mastro, bandeira])
    ruido = np.random.normal(0, 0.05, pontos)
    return padrao + ruido

# Visualizando todos os padrões que criamos
plt.figure(figsize=(15, 6))
plt.subplot(2, 3, 1); plt.plot(gerar_head_and_shoulders()); plt.title("Ombro-Cabeça-Ombro")
plt.subplot(2, 3, 2); plt.plot(gerar_double_top()); plt.title("Topo Duplo")
plt.subplot(2, 3, 3); plt.plot(gerar_no_pattern()); plt.title("Sem Padrão")
plt.subplot(2, 3, 4); plt.plot(gerar_bull_flag()); plt.title("Bandeira de Alta")
plt.subplot(2, 3, 5); plt.plot(gerar_bear_flag()); plt.title("Bandeira de Baixa")
plt.tight_layout()
plt.show()

# --- ESTÁGIO 2: PREPARAÇÃO DO DATASET (ATUALIZADO) ---

X = []
y = []
# Mapeamento de etiquetas atualizado para 5 classes
mapa_etiquetas = {
    0: 'Ombro-Cabeça-Ombro',
    1: 'Topo Duplo',
    2: 'Sem Padrão',
    3: 'Bandeira de Alta',
    4: 'Bandeira de Baixa'
}
num_classes = len(mapa_etiquetas) # Agora são 5 classes

num_amostras_por_classe = 500

print("\nGerando dataset de treinamento com 5 padrões...")
for _ in range(num_amostras_por_classe):
    X.append(gerar_head_and_shoulders())
    y.append(0)
    X.append(gerar_double_top())
    y.append(1)
    X.append(gerar_no_pattern())
    y.append(2)
    # Adicionando os novos padrões ao dataset
    X.append(gerar_bull_flag())
    y.append(3)
    X.append(gerar_bear_flag())
    y.append(4)

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)
# One-Hot Encoding de y atualizado para 5 classes
y = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dataset pronto: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste.")


# --- ESTÁGIO 3: CONSTRUÇÃO DO MODELO (ATUALIZADO) ---

print("\nConstruindo o modelo LSTM para 5 padrões...")
model = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    # A CAMADA DE SAÍDA AGORA TEM 5 NEURÔNIOS, UM PARA CADA PADRÃO
    Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- ESTÁGIO 4: TREINAMENTO DO MODELO ---
print("\nIniciando o treinamento do novo modelo...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- ESTÁGIO 5: AVALIAÇÃO E SALVAMENTO ---
print("\nIniciando a avaliação final...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%')

print("\nSalvando o novo modelo treinado...")
model.save('meu_modelo_de_padroes.keras')
print("✅ Novo modelo salvo com sucesso!")