# 1. Adicione estas duas linhas no topo de tudo
from agno.tools.googlesearch import GoogleSearchTools
from coingeckoToolKit import CoinGeckoToolKit
from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage
from agno.tools.reasoning import ReasoningTools
from agno.models.openrouter import OpenRouter
from agno.agent import Agent
from dotenv import load_dotenv
load_dotenv()


agent_storage: str = "tmp/agents.db"

reasoning_agent = Agent(
    model=OpenRouter(id="anthropic/claude-3.5-haiku"),
    tools=[
        ReasoningTools(add_instructions=True),
        GoogleSearchTools(),
        CoinGeckoToolKit()
    ],
    instructions=[
        # Search and Discovery Instructions
        "Use GoogleSearchTools to search for cryptocurrency tickers, names, and general information when user mentions a coin by name or symbol",
        "Use get_coins_list() from CoinGeckoToolKit to find the correct coin_id when you're unsure about the exact CoinGecko identifier",
        "Use get_trending() from CoinGeckoToolKit to discover what cryptocurrencies are currently popular and trending",
        
        # Market Data Instructions
        "Use get_market_data() from CoinGeckoToolKit to get current price, 24h change, market cap, and volume for any cryptocurrency",
        "Use get_coin_data() from CoinGeckoToolKit to get comprehensive information including description, website, market cap rank, and complete details about a cryptocurrency",
        
        # Historical Data Instructions  
        "Use get_coin_history() from CoinGeckoToolKit to get historical market data for a specific date (format: DD-MM-YYYY)",
        "Use get_coin_chart() from CoinGeckoToolKit to get price trend analysis over different time periods (1, 7, 14, 30, 90, 180, 365 days or max)",
        "Use get_coin_ohlc() from CoinGeckoToolKit to get candlestick data (Open, High, Low, Close) for technical analysis",
        
        # Analysis and Reasoning Instructions
        "Use ReasoningTools to analyze price trends, compare cryptocurrencies, interpret market data, and provide investment insights",
        "Always convert cryptocurrency prices to user's preferred currency when possible (USD, EUR, BRL, etc.)",
        "When analyzing trends, compare current data with historical data to provide meaningful insights",
        "Provide context about market cap rank, trading volume, and price changes to help users understand the cryptocurrency's position",
        
        # Response Guidelines
        "Always format cryptocurrency data clearly with emojis and proper currency symbols",
        "When user asks about multiple cryptocurrencies, gather data for all of them before providing analysis",
        "If a user asks about price predictions or investment advice, remind them that you provide data analysis but not financial advice",
        "For technical analysis requests, use OHLC data and explain the patterns in simple terms",
        "When showing historical data, always mention the time period and currency for context",
    ],
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
