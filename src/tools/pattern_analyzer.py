import numpy as np
import pandas as pd
import ccxt
import tensorflow as tf

# --- CONFIGURAÇÃO E CARREGAMENTO DO MODELO ---
# Esta parte roda uma vez quando o módulo é importado
try:
    MODELO_DE_PADROES = tf.keras.models.load_model('data/models/melhor_modelo.keras') # Ajustei o caminho
    MAPA_ETIQUETAS = {
        0: 'Ombro-Cabeça-Ombro', 1: 'Topo Duplo', 2: 'Sem Padrão',
        3: 'Bandeira de Alta', 4: 'Bandeira de Baixa'
    }
    print("✅ [Pattern Analyzer] Modelo de reconhecimento de padrões carregado.")
except Exception as e:
    print(f"🚨 [Pattern Analyzer] Erro Crítico: Não foi possível carregar o modelo.")
    print(f"   Erro: {e}")
    # Em uma aplicação real, você poderia decidir se quer parar o programa ou continuar sem a ferramenta.
    MODELO_DE_PADROES = None


# --- A FERRAMENTA ESPECIALISTA ---

def analisar_padrao_grafico(ticker: str, timeframe: str = '1d', limit: int = 100) -> str:
    """
    Analisa os dados de mercado de uma criptomoeda para identificar padrões gráficos técnicos.
    Use esta função sempre que o usuário pedir para 'analisar um gráfico', 'verificar padrões' 
    ou 'procurar por formações' em um ativo.
    """
    print(f"\n[🛠️ Ferramenta 'analisar_padrao_grafico' ativada para {ticker}]")
    
    if MODELO_DE_PADROES is None:
        return "Erro: O modelo de análise de padrões não está disponível."

    # Busca dados reais com os parâmetros dinâmicos
    print(f"Buscando dados para {ticker}...")
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
        if len(ohlcv) < limit:
            return f"Erro: Dados insuficientes ({len(ohlcv)} de {limit} velas) para {ticker} no timeframe {timeframe}."
        
        # Seleciona apenas os preços de fechamento
        dados_reais = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])['close'].values
        print("Dados obtidos com sucesso.")
    except Exception as e:
        return f"Erro ao buscar dados na exchange: {e}"

    # Prepara os dados (Normalização Min-Max)
    print("Normalizando dados para o modelo...")
    min_val = np.min(dados_reais)
    max_val = np.max(dados_reais)
    
    # Evita divisão por zero se todos os preços forem iguais
    if (max_val - min_val) == 0:
        return f"Não foi possível analisar {ticker}: os preços no período são constantes."

    dados_normalizados = (dados_reais - min_val) / (max_val - min_val)
    
    # Prepara o formato para o modelo LSTM: (1, timesteps, features)
    dados_para_previsao = dados_normalizados.reshape(1, limit, 1)

    # Faz a previsão com o modelo treinado
    print("Analisando com o modelo de padrões...")
    previsao = MODELO_DE_PADROES.predict(dados_para_previsao)
    classe_prevista = np.argmax(previsao[0])
    nome_padrao_previsto = MAPA_ETIQUETAS.get(classe_prevista, "Padrão Desconhecido")
    confianca = previsao[0][classe_prevista] * 100
    
    resultado_tecnico = f"Análise para {ticker} ({timeframe}, ultimas {limit} velas): Padrão detectado = '{nome_padrao_previsto}' (Confiança: {confianca:.2f}%)."
    print(f"[✅ Resultado da ferramenta]: {resultado_tecnico}")
    return resultado_tecnico


# --- BLOCO DE TESTE PARA EXECUÇÃO DIRETA DO ARQUIVO ---
if __name__ == "__main__":
    """
    Este bloco só é executado quando você roda 'python pattern_analyzer.py'
    diretamente no terminal. Ele serve para testar a função acima.
    """
    print("\n--- INICIANDO TESTES DO MÓDULO DE ANÁLISE DE PADRÕES ---")
    
    # Teste 1: BTC no gráfico diário
    resultado_btc = analisar_padrao_grafico(ticker='BTC/USDT', timeframe='1d', limit=100)
    print("-" * 50)

    # Teste 2: ETH no gráfico de 4 horas com mais dados
    resultado_eth = analisar_padrao_grafico(ticker='ETH/USDT', timeframe='4h', limit=120)
    print("-" * 50)

    # Teste 3: Ativo que pode não existir para testar o erro
    resultado_erro = analisar_padrao_grafico(ticker='FAKECOIN/USDT')
    print("-" * 50)