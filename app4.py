import json
import numpy as np
import pandas as pd
import ccxt
import tensorflow as tf
from openai import OpenAI

MODEL_FILE_PATH = 'meu_modelo_de_padroes.keras'

# --- CONFIGURAÇÃO E CARREGAMENTO INICIAL (sem alterações) ---
try:
    MODELO_DE_PADROES = tf.keras.models.load_model(MODEL_FILE_PATH)
    MAPA_ETIQUETAS = {
        0: 'Ombro-Cabeça-Ombro', 1: 'Topo Duplo', 2: 'Sem Padrão',
        3: 'Bandeira de Alta', 4: 'Bandeira de Baixa'
    }
    print("🧠 Modelo de reconhecimento de padrões carregado com sucesso.")
except Exception as e:
    print(f"🚨 Erro Crítico: Não foi possível carregar o arquivo {MODEL_FILE_PATH}.")
    print("Certifique-se de que o modelo foi treinado e o arquivo está na mesma pasta.")
    exit()

try:
    CLIENTE_OPENROUTER = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )
    print("🔌 Conexão com o OpenRouter estabelecida.")
except TypeError:
    print("🚨 Erro Crítico: A variável de ambiente OPENROUTER_API_KEY não foi encontrada.")
    exit()

# --- FERRAMENTAS ESPECIALISTAS QUE A LLM PODE USAR ---

# 1. ATUALIZAÇÃO DA FUNÇÃO-FERRAMENTA
def analisar_padrao_grafico(ticker: str, timeframe: str = '1d', limit: int = 100):
    """
    Analisa os dados de mercado de uma criptomoeda para identificar padrões gráficos técnicos.
    Use esta função sempre que o usuário pedir para 'analisar', 'verificar padrões' ou 'procurar por formações' em um gráfico.

    Args:
        ticker (str): O ticker do par a ser analisado, como 'BTC/USDT'.
        timeframe (str): O período de tempo de cada vela. Padrão '1d'. Exemplos: '1h', '4h', '1d', '1w'.
        limit (int): O número de velas a serem buscadas. Padrão 100.
    """
    print(f"\n[🛠️ Ferramenta 'analisar_padrao_grafico' ativada para {ticker}, timeframe: {timeframe}, limit: {limit}]")
    
    # Busca dados reais com os parâmetros dinâmicos
    print(f"Buscando dados reais para {ticker}...")
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
        if len(ohlcv) < limit:
            return f"Erro: Não foram encontrados dados suficientes ({len(ohlcv)} de {limit} períodos) para {ticker}."
        dados_reais = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])['close'].values
        print("Dados obtidos.")
    except Exception as e:
        return f"Erro ao buscar dados no ccxt: {e}"

    # Prepara os dados (Normalização)
    print("Normalizando e preparando os dados para análise...")
    min_val = np.min(dados_reais)
    max_val = np.max(dados_reais)
    dados_normalizados = (dados_reais - min_val) / (max_val - min_val)
    # Reshape dinâmico baseado no limite
    dados_para_previsao = dados_normalizados.reshape(1, limit, 1)

    # Faz a previsão com o modelo treinado
    # ATENÇÃO: Nosso modelo foi treinado com janelas de 100. Se o 'limit' for diferente, a precisão pode cair.
    # Em um projeto real, treinaríamos modelos para diferentes tamanhos de janela.
    print("Analisando com o modelo de padrões...")
    previsao = MODELO_DE_PADROES.predict(dados_para_previsao)
    classe_prevista = np.argmax(previsao[0])
    nome_padrao_previsto = MAPA_ETIQUETAS[classe_prevista]
    confianca = previsao[0][classe_prevista] * 100
    
    resultado_tecnico = f"Análise para {ticker} ({timeframe}, {limit} períodos) concluída. Padrão detectado: '{nome_padrao_previsto}' com {confianca:.2f}% de confiança."
    print(f"[✅ Resultado da ferramenta]: {resultado_tecnico}")
    return resultado_tecnico

# --- O AGENTE ORQUESTRADOR (LLM) ---

def agente_completo():
    """Função principal que gerencia a conversa e as ferramentas."""

    # 2. ATUALIZAÇÃO DO "MANUAL DE INSTRUÇÕES" PARA A LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analisar_padrao_grafico",
                "description": "Analisa o gráfico de uma criptomoeda em busca de padrões técnicos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "O ticker do par a ser analisado, ex: 'BTC/USDT', 'ETH/BRL'"
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "O período de cada vela. Exemplos: '1h' (hora), '4h' (4 horas), '1d' (dia), '1w' (semana). O padrão é '1d'."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "O número de velas (períodos) a serem analisados. O padrão é 100."
                        }
                    },
                    "required": ["ticker"], # Apenas o ticker é estritamente obrigatório
                },
            },
        }
    ]
    
    messages = [{"role": "system", "content": "Você é um assistente de análise de criptomoedas. Sua função é extrair o ativo, o timeframe e o limite de períodos do pedido do usuário e usar a ferramenta 'analisar_padrao_grafico'. Depois, apresente o resultado ao usuário."}]

    print("\n--- 🤖 Agente de Análise Técnica Avançado ---")
    print("Estou pronto. Peça para analisar um ativo, especificando timeframe ou períodos se desejar.")
    print("Para sair, digite 'sair'.")

    while True:
        entrada_usuario = input("\n> ")
        if not entrada_usuario.strip(): continue
        if entrada_usuario.lower() in ['sair', 'exit']: 
            print("Até Logo!")
            break
        
        messages.append({"role": "user", "content": entrada_usuario})

        try:
            response = CLIENTE_OPENROUTER.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"[🤖 LLM decidiu usar a ferramenta: '{function_name}' com os argumentos: {function_args}]")
                
                # 3. ATUALIZAÇÃO NA CHAMADA DA FERRAMENTA
                # Usamos **function_args para passar todos os argumentos de uma vez.
                # A função Python usará os defaults se algum não for fornecido pela LLM.
                function_response = analisar_padrao_grafico(**function_args)
                
                messages.append(response_message)
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})
                
                print("\n[🤖 LLM está gerando a resposta final...]")
                final_response = CLIENTE_OPENROUTER.chat.completions.create(model="anthropic/claude-3-haiku", messages=messages)
                print(f"\nAgente diz: {final_response.choices[0].message.content}")
                messages.append(final_response.choices[0].message)
            else:
                print(f"\nAgente diz: {response_message.content}")
                messages.append(response_message)
        except Exception as e:
            print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    agente_completo()