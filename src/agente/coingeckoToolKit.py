import requests
import os
from typing import Any, Dict, List, Optional
from agno.tools import Toolkit
from dotenv import load_dotenv
load_dotenv()


class CoinGeckoToolKit(Toolkit):
    """
    CoinGecko Toolkit for fetching cryptocurrency market data.
    This toolkit provides tools to get market information like price, volume, 
    and market cap for cryptocurrencies through the CoinGecko API.

    Two modes available:
    1. Direct mode: Uses API key directly (COINGECKO_API_KEY)
    2. Proxy mode: Uses proxy server that handles API authentication (COINGECKO_PROXY_URL)

    Args:
        timeout (Optional[int]): Timeout for HTTP requests, default is 10 seconds.
        use_proxy (Optional[bool]): If True, uses proxy mode. If False, uses direct mode.
                                   If None, auto-detects based on environment variables.
    """

    def __init__(self, timeout: Optional[int] = 10, use_proxy: Optional[bool] = None):
        super().__init__(name="coingecko_tools")
        self.timeout: Optional[int] = timeout

        # Auto-detect mode based on environment variables if not specified
        if use_proxy is None:
            self.use_proxy = bool(os.getenv("COINGECKO_PROXY_URL"))
        else:
            self.use_proxy = use_proxy

        if self.use_proxy:
            self.proxy_server_url = os.getenv(
                "COINGECKO_PROXY_URL", "http://localhost:8000")
            self.api_key = None
        else:
            self.proxy_server_url = None
            self.api_key = os.getenv("COINGECKO_API_KEY")

        self.register(self.get_market_data)

    def get_market_data(self, coin_id: str, vs_currency: str = "usd") -> str:
        """
        Get market data for a specific cryptocurrency including price, volume and market cap.

        In direct mode: Calls CoinGecko API directly using API key.
        In proxy mode: Calls a proxy server that queries the CoinGecko /coins/markets endpoint.

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currency (str): The target currency for price data (e.g., "usd", "eur", "brl"). 
                              Default is "usd".

        Returns:
            str: A formatted string containing the market data (price, volume, market cap) 
                 or an error message if the request fails.
        """
        print(
            f"🎯 [DEBUG] get_market_data CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}'")

        try:
            if self.use_proxy:
                return self._get_market_data_via_proxy(coin_id, vs_currency)
            else:
                return self._get_market_data_direct(coin_id, vs_currency)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching market data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting market data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _get_market_data_direct(self, coin_id: str, vs_currency: str) -> str:
        """
        Get market data directly from CoinGecko API using API key.
        """
        if not self.api_key:
            error_msg = "Missing COINGECKO_API_KEY environment variable. Please set your CoinGecko API key."
            print(f"❌ [DEBUG] {error_msg}")
            return error_msg

        # CoinGecko Pro API endpoint (required for Pro API keys)
        api_endpoint = "https://pro-api.coingecko.com/api/v3/coins/markets"

        # Parameters for the CoinGecko markets endpoint - restored full functionality
        params = {
            "ids": coin_id,
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": 1,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }

        # Headers with API key - restored full functionality
        headers = {
            "x-cg-pro-api-key": self.api_key,
            "User-Agent": "AGNO-CoinGecko-Toolkit/1.0",
            "Accept": "application/json"
        }

        # Make the request directly to CoinGecko Pro API
        response = requests.get(
            api_endpoint, params=params, headers=headers, timeout=self.timeout)

        print(f"📊 [DEBUG] Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ [DEBUG] Resposta de erro: {response.text}")

        response.raise_for_status()

        response_data = response.json()
        return self._format_market_data_response(response_data, coin_id, vs_currency)

    def _get_market_data_via_proxy(self, coin_id: str, vs_currency: str) -> str:
        """
        Get market data via proxy server that handles CoinGecko API authentication.
        """
        # Construct the proxy server endpoint URL
        proxy_endpoint = f"{self.proxy_server_url}/api/coingecko/markets"

        # Parameters for the CoinGecko markets endpoint - restored full functionality
        params = {
            "ids": coin_id,
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": 1,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }

        # Make the request to the proxy server
        response = requests.get(
            proxy_endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()

        return self._format_market_data_response(response.json(), coin_id, vs_currency)

    def _format_market_data_response(self, market_data: List[Dict], coin_id: str, vs_currency: str) -> str:
        """
        Format the market data response into a readable string.
        """
        if not market_data or len(market_data) == 0:
            no_data_msg = f"No market data found for {coin_id}"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        # Extract data from the first (and only) result
        coin_data = market_data[0]

        # Format the response
        price = coin_data.get('current_price', 'N/A')
        market_cap = coin_data.get('market_cap', 'N/A')
        volume_24h = coin_data.get('total_volume', 'N/A')
        price_change_24h = coin_data.get('price_change_percentage_24h', 'N/A')
        symbol = coin_data.get('symbol', '').upper()
        name = coin_data.get('name', coin_id)

        # Format numbers for better readability
        if isinstance(price, (int, float)):
            price_formatted = f"{price:,.2f}" if price >= 0.01 else f"{price:.6f}"
        else:
            price_formatted = str(price)

        if isinstance(market_cap, (int, float)):
            market_cap_formatted = f"${market_cap:,.0f}"
        else:
            market_cap_formatted = str(market_cap)

        if isinstance(volume_24h, (int, float)):
            volume_formatted = f"${volume_24h:,.0f}"
        else:
            volume_formatted = str(volume_24h)

        if isinstance(price_change_24h, (int, float)):
            change_formatted = f"{price_change_24h:+.2f}%"
        else:
            change_formatted = str(price_change_24h)

        result = f"""
🪙 **{name} ({symbol})**
💰 **Price**: {price_formatted} {vs_currency.upper()}
📊 **24h Change**: {change_formatted}
🏦 **Market Cap**: {market_cap_formatted}
📈 **24h Volume**: {volume_formatted}
        """.strip()

        print(f"✅ [DEBUG] Dados formatados com sucesso para {coin_id}")
        return result
