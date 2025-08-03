# 1. Adicione estas duas linhas no topo de tudo
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.webtools import WebTools
from agno.storage.sqlite import SqliteStorage

agent_storage: str = "tmp/agents.db"

reasoning_agent = Agent(
    model=Gemini(id="gemini-1.5-flash-latest"),
    tools=[
        ReasoningTools(add_instructions=True),
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True
        ),
        WebTools()
    ],
    instructions="search in web tickers to the stock or coin name",
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

if __name__ == "__main__":
    print("--- Agente com Acesso à Web ---")
    print("Digite uma pergunta ou 'sair' para terminar.\n")

    while True:
        try:
            user_input = input("Você: ")
            if user_input.lower() == 'sair':
                print("Encerrando...")
                break
            
            print("Agente está pensando...")
            response = reasoning_agent.run(user_input)
            
            print("\n--- Resposta do Agente ---")
            print(response.content)
            print("--------------------------\n")

        except Exception as e:
            print(f"Ocorreu um erro: {e}")