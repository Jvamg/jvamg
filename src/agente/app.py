# 1. Adicione estas duas linhas no topo de tudo
from agno.tools.googlesearch import GoogleSearchTools
from coingeckoToolKit import CoinGeckoToolKit
from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.models.google import Gemini
from agno.agent import Agent
from dotenv import load_dotenv
load_dotenv()


agent_storage: str = "tmp/agents.db"

reasoning_agent = Agent(
    model=Gemini(id="gemini-1.5-pro"),
    tools=[
        ReasoningTools(add_instructions=True),
        GoogleSearchTools(),
        CoinGeckoToolKit()
    ],
    instructions=[
        "Use GoogleSearchTools to get tickers to the Cryptocurrency",
        "Use CoinGeckoToolKit to get information of the Cryptocurrency",
        "Use ReasoningTools to reason about the information",],
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

playground_app = Playground(agents=[reasoning_agent])
app = playground_app.get_app()

if __name__ == "__main__":
    playground_app.serve("app:app", reload=True)
