# Arquivo: src/tools/quantitative_analyzer.py

import ccxt
import pandas as pd
import pandas_ta as ta

class AnalistaQuantitativo:
    def __init__(self, exchange_id='binance'):
        """
        Inicializa o analista conectando-se à exchange desejada.
        """
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class()
            print(f"Conectado à exchange: {self.exchange.name}")
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' não encontrada.")

    def fetch_data(self, ticker='BTC/USDT', timeframe='1d', limit=200):
        """
        Busca os dados OHLCV e os retorna como um DataFrame do Pandas.
        """
        try:
            print(f"Buscando {limit} velas de {ticker} no timeframe de {timeframe}...")
            ohlcv = self.exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True) # Boa prática: usar o tempo como índice
            return df
        except Exception as e:
            print(f"Erro ao buscar dados para {ticker}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame):
        """
        Calcula um conjunto pré-definido de indicadores técnicos no DataFrame.
        """
        if df is None or df.empty:
            return None
        
        # Dentro do método calculate_indicators:

        print("Calculando indicadores selecionados...")
        # Adiciona explicitamente cada indicador que queremos
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True) # Bandas de Bollinger
        df.ta.adx(append=True)    # ADX
        df.ta.obv(append=True)     # OBV

        # Adicionando duas médias móveis exponenciais para análise de tendência
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        
        return df

    def get_full_analysis(self, ticker='BTC/USDT', timeframe='1d'):
        """
        Orquestra o processo: busca os dados, calcula os indicadores e 
        retorna os valores mais recentes.
        """
        df_data = self.fetch_data(ticker, timeframe)
        if df_data is None:
            return {"error": "Não foi possível obter os dados."}
            
        df_with_indicators = self.calculate_indicators(df_data)
        if df_with_indicators is None:
            return {"error": "Não foi possível calcular os indicadores."}

        # Pega a última linha com os dados mais recentes
        last_row = df_with_indicators.iloc[-1]
        
        # Monta um JSON de resposta limpo e organizado
        resultado = {
            'ticker': ticker,
            'timestamp': last_row.name.isoformat(),
            'close': last_row.get('close'),
            'RSI_14': last_row.get('RSI_14'),
            'MACD_12_26_9': last_row.get('MACD_12_26_9'),
            'BBL_20_2.0': last_row.get('BBL_20_2.0'), # Banda de Bollinger Inferior
            'BBU_20_2.0': last_row.get('BBU_20_2.0'), # Banda de Bollinger Superior
            'ADX_14': last_row.get('ADX_14'),
            'OBV': last_row.get('OBV')
        }
        # Remove chaves com valores nulos (indicadores que precisam de mais dados para começar)
        return {k: v for k, v in resultado.items() if pd.notna(v)}

# --- Bloco de Teste ---
if __name__ == '__main__':
    # 1. Cria uma instância do nosso analista
    analista = AnalistaQuantitativo()
    
    # 2. Roda a análise completa para o BTC no gráfico diário
    analise_btc = analista.get_full_analysis(ticker='BTC/USDT', timeframe='1d')
    
    if analise_btc:
        print("\n--- Análise Quantitativa para BTC/USDT (1d) ---")
        for key, value in analise_btc.items():
            # Formata os números para melhor leitura
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")