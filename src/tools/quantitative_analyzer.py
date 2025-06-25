import ccxt
import pandas as pd
import pandas_ta as ta

def obter_indicadores(ticker='BTC/USDT', timeframe='1d', limit=200):
    """
    Busca os dados de mercado mais recentes para um ticker e calcula 
    os principais indicadores técnicos.
    """
    try:
        # 1. Conectar-se à exchange (a Binance não exige chave de API para dados públicos)
        exchange = ccxt.binance()

        # 2. Buscar os dados OHLCV (Open, High, Low, Close, Volume)
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)

        # 3. Converter para um DataFrame do Pandas
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Converte o tempo

        # 4. Usar o pandas-ta para calcular os indicadores e adicioná-los ao DataFrame
        df.ta.rsi(append=True)  # Adiciona a coluna 'RSI_14'
        df.ta.macd(append=True) # Adiciona as colunas 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'

        # 5. Pegar apenas os dados mais recentes (a última linha)
        dados_recentes = df.iloc[-1]

        # 6. Retornar um JSON limpo
        resultado = {
            'ticker': ticker,
            'timestamp': dados_recentes['timestamp'].isoformat(),
            'close': dados_recentes['close'],
            'rsi_14': dados_recentes['RSI_14'],
            'macd': dados_recentes['MACD_12_26_9']
        }

        return resultado

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None

# Bloco de teste
if __name__ == '__main__':
    indicadores_btc = obter_indicadores('BTC/USDT', timeframe='4h')
    if indicadores_btc:
        print("Indicadores Atuais para BTC/USDT (4h):")
        print(indicadores_btc)