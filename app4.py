import os
import json
import numpy as np
import pandas as pd
import ccxt
import tensorflow as tf
from openai import OpenAI
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO E CARREGAMENTO INICIAL ---

# Carregando o cérebro do nosso analista especialista uma única vez
# --- CONFIGURAÇÃO E CARREGAMENTO INICIAL ---
try:
    MODELO_DE_PADROES = tf.keras.models.load_model('meu_modelo_de_padroes.keras')
    # ATUALIZE ESTE DICIONÁRIO PARA CONTER OS 5 PADRÕES
    MAPA_ETIQUETAS = {
        0: 'Ombro-Cabeça-Ombro',
        1: 'Topo Duplo',
        2: 'Sem Padrão',
        3: 'Bandeira de Alta',  # <-- Adicionado
        4: 'Bandeira de Baixa'  # <-- Adicionado
    }
    print("🧠 Modelo de reconhecimento de padrões carregado com sucesso.")
except Exception as e:
    print("🚨 Erro Crítico: Não foi possível carregar o arquivo 'meu_modelo_de_padroes.keras'.")
    print("Certifique-se de que o modelo foi treinado e o arquivo está na mesma pasta.")
    exit()

# Configuração do cliente do OpenRouter
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


def analisar_padrao_grafico(ticker: str):
    """
    Analisa os dados de mercado mais recentes de uma criptomoeda para identificar padrões gráficos técnicos.
    Use esta função sempre que o usuário pedir para 'analisar', 'verificar padrões' ou 'procurar por formações' em um gráfico.

    Args:
        ticker (str): O ticker do par a ser analisado, como 'BTC/USDT'.
    """
    print(f"\n[🛠️ Ferramenta 'analisar_padrao_grafico' ativada para {ticker}]")

    # 1. Buscar dados reais
    print(f"Buscando dados reais para {ticker}...")
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(ticker, '1d', limit=100)
        if len(ohlcv) < 100:
            return f"Erro: Não foram encontrados dados suficientes (mínimo 100 dias) para {ticker}."
        dados_reais = pd.DataFrame(ohlcv, columns=[
                                   'timestamp', 'open', 'high', 'low', 'close', 'volume'])['close'].values
        print("Dados obtidos.")
    except Exception as e:
        return f"Erro ao buscar dados no ccxt: {e}"

    # 2. Preparar os dados para o modelo
    print("Normalizando e preparando os dados para análise...")
    min_val = np.min(dados_reais)
    max_val = np.max(dados_reais)
    dados_normalizados = (dados_reais - min_val) / (max_val - min_val)
    dados_para_previsao = dados_normalizados.reshape(1, 100, 1)

    # 3. Fazer a previsão com o modelo treinado
    print("Analisando com o modelo de padrões...")
    previsao = MODELO_DE_PADROES.predict(dados_para_previsao)
    classe_prevista = np.argmax(previsao[0])
    nome_padrao_previsto = MAPA_ETIQUETAS[classe_prevista]
    confianca = previsao[0][classe_prevista] * 100

    resultado = f"Análise para {ticker} concluída. Padrão detectado: '{nome_padrao_previsto}' com {confianca:.2f}% de confiança."
    print(f"[✅ Resultado da ferramenta]: {resultado}")
    return resultado

# --- O AGENTE ORQUESTRADOR (LLM) ---


# --- O AGENTE ORQUESTRADOR (LLM) - VERSÃO CORRIGIDA ---

def agente_completo():
    """Função principal que gerencia a conversa e as ferramentas."""

    # Descrevemos a ferramenta para a LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analisar_padrao_grafico",
                "description": "Analisa o gráfico de uma criptomoeda em busca de padrões técnicos.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string", "description": "O ticker do par, ex: 'BTC/USDT'"}},
                    "required": ["ticker"],
                },
            },
        }
    ]

    print("\n--- 🤖 Agente de Análise Técnica Completo ---")
    print("Estou pronto. Peça para analisar um ativo (ex: 'analise o bitcoin pra mim'). Para sair, digite 'sair'.")

    # A principal mudança está neste loop.
    while True:
        entrada_usuario = input("\n> ")
        if not entrada_usuario.strip():  # Ignora se o usuário apertar Enter sem digitar nada
            continue

        if entrada_usuario.lower() in ['sair', 'exit']:
            print("Até logo!")
            break

        # Criamos a lista de mensagens aqui, garantindo que a primeira é sempre do usuário
        messages = [{"role": "user", "content": entrada_usuario}]

        try:
            # Envia para o OpenRouter
            response = CLIENTE_OPENROUTER.chat.completions.create(
                # Verifique se o nome do modelo está correto.
                # Se estiver usando um modelo do Google, ele deve ser algo como "google/gemini-pro-1.5"
                model="anthropic/claude-3-haiku",  # Um modelo robusto que aceita role:system
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message

            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"[🤖 LLM decidiu usar a ferramenta: '{function_name}']")

                # Executa a ferramenta
                function_response = analisar_padrao_grafico(ticker=function_args.get("ticker"))

                # Adiciona a chamada da ferramenta e a resposta da ferramenta ao histórico
                messages.append(response_message)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

                # Envia o histórico completo de volta para a LLM gerar uma resposta final
                print("\n[🤖 LLM está gerando a resposta final...]")
                final_response = CLIENTE_OPENROUTER.chat.completions.create(
                    model="anthropic/claude-3-haiku",
                    messages=messages
                )
                print(f"\nAgente diz: {final_response.choices[0].message.content}")
            else:
                # Se não usou ferramenta, apenas responde
                print(f"\nAgente diz: {response_message.content}")

        except Exception as e:
            print(f"Ocorreu um erro ao chamar a API: {e}")


if __name__ == "__main__":
    agente_completo()
