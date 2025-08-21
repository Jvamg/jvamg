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
        self.register(self.get_coin_data)
        self.register(self.get_coin_history)
        self.register(self.get_coin_chart)
        self.register(self.get_coin_ohlc)
        self.register(self.get_trending)
        self.register(self.get_coins_list)

    def _make_request(self, endpoint_path: str, params: Optional[Dict] = None) -> Dict:
        """
        Generic method to make requests to CoinGecko API in both direct and proxy modes.
        
        Args:
            endpoint_path (str): The API endpoint path (e.g., "coins/markets", "coins/bitcoin")
            params (Optional[Dict]): Query parameters for the request
        
        Returns:
            Dict: JSON response from the API
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        if params is None:
            params = {}
            
        if self.use_proxy:
            # Construct proxy server endpoint URL
            url = f"{self.proxy_server_url}/api/coingecko/{endpoint_path}"
            headers = {"User-Agent": "AGNO-CoinGecko-Toolkit/1.0", "Accept": "application/json"}
        else:
            # Construct direct CoinGecko Pro API endpoint URL
            if not self.api_key:
                raise ValueError("Missing COINGECKO_API_KEY environment variable. Please set your CoinGecko API key.")
            
            url = f"https://pro-api.coingecko.com/api/v3/{endpoint_path}"
            headers = {
                "x-cg-pro-api-key": self.api_key,
                "User-Agent": "AGNO-CoinGecko-Toolkit/1.0",
                "Accept": "application/json"
            }
        
        # Make the request
        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        
        print(f"📊 [DEBUG] Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ [DEBUG] Resposta de erro: {response.text}")
        
        response.raise_for_status()
        return response.json()

    def get_market_data(self, coin_id: str, vs_currency: str = "usd") -> str:
        """
        Get market data for a specific cryptocurrency including price, volume and market cap.

        Uses either direct API key access or proxy server based on configuration.

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currency (str): The target currency for price data (e.g., "usd", "eur", "brl"). 
                              Default is "usd".

        Returns:
            str: A formatted string containing the market data (price, volume, market cap) 
                 or an error message if the request fails.
        """
        print(f"🎯 [DEBUG] get_market_data CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}'")

        try:
            # Parameters for the CoinGecko markets endpoint
            params = {
                "ids": coin_id,
                "vs_currency": vs_currency,
                "order": "market_cap_desc",
                "per_page": 1,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h"
            }

            response_data = self._make_request("coins/markets", params)
            return self._format_market_data_response(response_data, coin_id, vs_currency)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching market data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting market data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

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

    def get_coin_data(self, coin_id: str, localization: bool = True, tickers: bool = True, 
                     market_data: bool = True, community_data: bool = True, 
                     developer_data: bool = True, sparkline: bool = False) -> str:
        """
        Get comprehensive data for a specific cryptocurrency by ID.

        This endpoint provides the most complete data set available for a cryptocurrency,
        including market data, community metrics, developer activity, and more.

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            localization (bool): Include localized language fields. Default is True.
            tickers (bool): Include tickers data. Default is True.
            market_data (bool): Include market data. Default is True.
            community_data (bool): Include community data. Default is True.
            developer_data (bool): Include developer data. Default is True.
            sparkline (bool): Include sparkline data. Default is False.

        Returns:
            str: A formatted string containing comprehensive coin data or an error message.
        """
        print(f"🎯 [DEBUG] get_coin_data CHAMADA! coin_id='{coin_id}'")

        try:
            # Parameters for the coin data endpoint - convert booleans to strings
            params = {
                "localization": str(localization).lower(),
                "tickers": str(tickers).lower(),
                "market_data": str(market_data).lower(),
                "community_data": str(community_data).lower(),
                "developer_data": str(developer_data).lower(),
                "sparkline": str(sparkline).lower()
            }

            response_data = self._make_request(f"coins/{coin_id}", params)
            return self._format_coin_data_response(response_data)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching coin data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting coin data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_coin_data_response(self, coin_data: Dict) -> str:
        """
        Format the comprehensive coin data response into a readable string.
        """
        if not coin_data:
            no_data_msg = "No coin data found"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        # Extract basic information
        name = coin_data.get('name', 'N/A')
        symbol = coin_data.get('symbol', '').upper()
        current_price = 'N/A'
        market_cap = 'N/A'
        market_cap_rank = coin_data.get('market_cap_rank', 'N/A')
        
        # Extract market data if available
        if 'market_data' in coin_data and coin_data['market_data']:
            market_info = coin_data['market_data']
            if 'current_price' in market_info and 'usd' in market_info['current_price']:
                price = market_info['current_price']['usd']
                if isinstance(price, (int, float)):
                    current_price = f"${price:,.2f}" if price >= 0.01 else f"${price:.6f}"
            
            if 'market_cap' in market_info and 'usd' in market_info['market_cap']:
                cap = market_info['market_cap']['usd']
                if isinstance(cap, (int, float)):
                    market_cap = f"${cap:,.0f}"

        # Extract description
        description = ""
        if 'description' in coin_data and 'en' in coin_data['description']:
            desc_text = coin_data['description']['en']
            if desc_text:
                # Truncate description to first 200 characters
                description = desc_text[:200] + "..." if len(desc_text) > 200 else desc_text

        # Extract links
        homepage = ""
        if 'links' in coin_data and 'homepage' in coin_data['links']:
            homepages = coin_data['links']['homepage']
            if homepages and len(homepages) > 0 and homepages[0]:
                homepage = homepages[0]

        result = f"""
🪙 **{name} ({symbol})**
💰 **Current Price**: {current_price}
🏦 **Market Cap**: {market_cap}
📊 **Market Cap Rank**: #{market_cap_rank}
🌐 **Homepage**: {homepage if homepage else 'N/A'}
📝 **Description**: {description if description else 'N/A'}
        """.strip()

        print(f"✅ [DEBUG] Dados completos formatados com sucesso para {coin_data.get('id', 'unknown')}")
        return result

    def get_coin_history(self, coin_id: str, date: str, localization: bool = False) -> str:
        """
        Get historical market data for a specific cryptocurrency on a given date.

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            date (str): Date in format DD-MM-YYYY (e.g., "15-05-2022").
            localization (bool): Include localized language fields. Default is False.

        Returns:
            str: A formatted string containing historical market data or an error message.
        """
        print(f"🎯 [DEBUG] get_coin_history CHAMADA! coin_id='{coin_id}', date='{date}'")

        try:
            # Parameters for the historical data endpoint - convert boolean to string
            params = {
                "date": date,
                "localization": str(localization).lower()
            }

            response_data = self._make_request(f"coins/{coin_id}/history", params)
            return self._format_coin_history_response(response_data, date)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching historical data for {coin_id} on {date}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting historical data for {coin_id} on {date}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_coin_history_response(self, history_data: Dict, date: str) -> str:
        """
        Format the historical coin data response into a readable string.
        """
        if not history_data:
            no_data_msg = f"No historical data found for date {date}"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        # Extract basic information
        name = history_data.get('name', 'N/A')
        symbol = history_data.get('symbol', '').upper()
        
        # Extract market data
        current_price = 'N/A'
        market_cap = 'N/A'
        total_volume = 'N/A'
        
        if 'market_data' in history_data and history_data['market_data']:
            market_info = history_data['market_data']
            
            if 'current_price' in market_info and 'usd' in market_info['current_price']:
                price = market_info['current_price']['usd']
                if isinstance(price, (int, float)):
                    current_price = f"${price:,.2f}" if price >= 0.01 else f"${price:.6f}"
            
            if 'market_cap' in market_info and 'usd' in market_info['market_cap']:
                cap = market_info['market_cap']['usd']
                if isinstance(cap, (int, float)):
                    market_cap = f"${cap:,.0f}"
                    
            if 'total_volume' in market_info and 'usd' in market_info['total_volume']:
                volume = market_info['total_volume']['usd']
                if isinstance(volume, (int, float)):
                    total_volume = f"${volume:,.0f}"

        result = f"""
📅 **Historical Data for {name} ({symbol}) - {date}**
💰 **Price**: {current_price}
🏦 **Market Cap**: {market_cap}
📈 **24h Volume**: {total_volume}
        """.strip()

        print(f"✅ [DEBUG] Dados históricos formatados com sucesso para {history_data.get('id', 'unknown')} em {date}")
        return result

    def get_coin_chart(self, coin_id: str, vs_currency: str = "usd", days: str = "1", 
                      interval: Optional[str] = None, precision: Optional[str] = None) -> str:
        """
        Get historical chart data for a cryptocurrency.

        This endpoint provides historical price, market cap, and volume data
        suitable for creating charts and analyzing trends over time.

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currency (str): The target currency for price data (e.g., "usd", "eur", "brl"). Default is "usd".
            days (str): Data up to number of days ago. Valid values: 1, 7, 14, 30, 90, 180, 365, max.
            interval (Optional[str]): Data interval. Possible values: daily. Leave empty for automatic granularity.
            precision (Optional[str]): Decimal place for currency price value. Valid values: 0 - 18.

        Returns:
            str: A formatted string containing chart data summary or an error message.
        """
        print(f"🎯 [DEBUG] get_coin_chart CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}', days='{days}'")

        try:
            # Parameters for the chart data endpoint
            params = {
                "vs_currency": vs_currency,
                "days": days
            }
            
            if interval:
                params["interval"] = interval
            if precision:
                params["precision"] = precision

            response_data = self._make_request(f"coins/{coin_id}/market_chart", params)
            return self._format_coin_chart_response(response_data, coin_id, vs_currency, days)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching chart data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting chart data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_coin_chart_response(self, chart_data: Dict, coin_id: str, vs_currency: str, days: str) -> str:
        """
        Format the chart data response into a readable string.
        """
        if not chart_data:
            no_data_msg = f"No chart data found for {coin_id}"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        # Extract data arrays
        prices = chart_data.get('prices', [])
        market_caps = chart_data.get('market_caps', [])
        total_volumes = chart_data.get('total_volumes', [])
        
        # Calculate basic statistics if we have data
        if prices and len(prices) > 0:
            # Get first and last prices for trend analysis
            first_price = prices[0][1]
            last_price = prices[-1][1]
            price_change = ((last_price - first_price) / first_price) * 100
            
            # Find highest and lowest prices
            price_values = [p[1] for p in prices]
            highest_price = max(price_values)
            lowest_price = min(price_values)
            
            # Format prices
            first_formatted = f"${first_price:,.2f}" if first_price >= 0.01 else f"${first_price:.6f}"
            last_formatted = f"${last_price:,.2f}" if last_price >= 0.01 else f"${last_price:.6f}"
            high_formatted = f"${highest_price:,.2f}" if highest_price >= 0.01 else f"${highest_price:.6f}"
            low_formatted = f"${lowest_price:,.2f}" if lowest_price >= 0.01 else f"${lowest_price:.6f}"
            change_formatted = f"{price_change:+.2f}%"
            
            result = f"""
📈 **Chart Data for {coin_id.title()} - Last {days} day(s)**
💰 **Period Start**: {first_formatted} {vs_currency.upper()}
💰 **Period End**: {last_formatted} {vs_currency.upper()}
📊 **Change**: {change_formatted}
📈 **Highest**: {high_formatted}
📉 **Lowest**: {low_formatted}
📅 **Data Points**: {len(prices)} price entries, {len(market_caps)} market cap entries, {len(total_volumes)} volume entries
            """.strip()
        else:
            result = f"No price data available for {coin_id} in the last {days} day(s)"

        print(f"✅ [DEBUG] Dados de gráfico formatados com sucesso para {coin_id}")
        return result

    def get_coin_ohlc(self, coin_id: str, vs_currency: str = "usd", days: str = "1", 
                     precision: Optional[str] = None) -> str:
        """
        Get OHLC (Open, High, Low, Close) data for a cryptocurrency.

        This endpoint provides candlestick data suitable for technical analysis.

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currency (str): The target currency for price data (e.g., "usd", "eur", "brl"). Default is "usd".
            days (str): Data up to number of days ago. Valid values: 1, 7, 14, 30, 90, 180, 365, max.
            precision (Optional[str]): Decimal place for currency price value. Valid values: 0 - 18.

        Returns:
            str: A formatted string containing OHLC data summary or an error message.
        """
        print(f"🎯 [DEBUG] get_coin_ohlc CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}', days='{days}'")

        try:
            # Parameters for the OHLC data endpoint
            params = {
                "vs_currency": vs_currency,
                "days": days
            }
            
            if precision:
                params["precision"] = precision

            response_data = self._make_request(f"coins/{coin_id}/ohlc", params)
            return self._format_coin_ohlc_response(response_data, coin_id, vs_currency, days)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching OHLC data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting OHLC data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_coin_ohlc_response(self, ohlc_data: List, coin_id: str, vs_currency: str, days: str) -> str:
        """
        Format the OHLC data response into a readable string.
        """
        if not ohlc_data or len(ohlc_data) == 0:
            no_data_msg = f"No OHLC data found for {coin_id}"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        # OHLC data comes as array of [timestamp, open, high, low, close]
        data_points = len(ohlc_data)
        
        if data_points > 0:
            # Get latest OHLC data (last entry)
            latest_data = ohlc_data[-1]
            timestamp, open_price, high_price, low_price, close_price = latest_data
            
            # Format prices
            open_formatted = f"${open_price:,.2f}" if open_price >= 0.01 else f"${open_price:.6f}"
            high_formatted = f"${high_price:,.2f}" if high_price >= 0.01 else f"${high_price:.6f}"
            low_formatted = f"${low_price:,.2f}" if low_price >= 0.01 else f"${low_price:.6f}"
            close_formatted = f"${close_price:,.2f}" if close_price >= 0.01 else f"${close_price:.6f}"
            
            # Calculate change percentage
            if open_price > 0:
                change_percent = ((close_price - open_price) / open_price) * 100
                change_formatted = f"{change_percent:+.2f}%"
            else:
                change_formatted = "N/A"

            result = f"""
🕯️ **OHLC Data for {coin_id.title()} - Last {days} day(s)**
🟢 **Open**: {open_formatted} {vs_currency.upper()}
📈 **High**: {high_formatted} {vs_currency.upper()}
📉 **Low**: {low_formatted} {vs_currency.upper()}
🔴 **Close**: {close_formatted} {vs_currency.upper()}
📊 **Change**: {change_formatted}
📅 **Data Points**: {data_points} candles available
            """.strip()
        else:
            result = f"No OHLC data available for {coin_id} in the last {days} day(s)"

        print(f"✅ [DEBUG] Dados OHLC formatados com sucesso para {coin_id}")
        return result

    def get_trending(self) -> str:
        """
        Get trending cryptocurrencies based on search activity.

        This endpoint provides the top 7 cryptocurrencies that are trending 
        on CoinGecko based on search activity in the last 24 hours.

        Returns:
            str: A formatted string containing trending cryptocurrencies or an error message.
        """
        print(f"🎯 [DEBUG] get_trending CHAMADA!")

        try:
            response_data = self._make_request("search/trending")
            return self._format_trending_response(response_data)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching trending data: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting trending data: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_trending_response(self, trending_data: Dict) -> str:
        """
        Format the trending data response into a readable string.
        """
        if not trending_data:
            no_data_msg = "No trending data found"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        coins = trending_data.get('coins', [])
        
        if not coins:
            result = "No trending coins found at the moment."
        else:
            result = "🔥 **Trending Cryptocurrencies (Top 7)**\n\n"
            
            for i, coin_info in enumerate(coins, 1):
                coin = coin_info.get('item', {})
                name = coin.get('name', 'N/A')
                symbol = coin.get('symbol', '').upper()
                market_cap_rank = coin.get('market_cap_rank', 'N/A')
                
                result += f"{i}. **{name} ({symbol})** - Rank #{market_cap_rank}\n"

        print(f"✅ [DEBUG] Dados de trending formatados com sucesso")
        return result

    def get_coins_list(self, include_platform: bool = False) -> str:
        """
        Get list of all supported cryptocurrencies.

        This endpoint provides a complete list of all cryptocurrencies supported 
        by CoinGecko with their IDs, symbols, and names.

        Args:
            include_platform (bool): Include platform contract addresses. Default is False.

        Returns:
            str: A formatted string containing information about available coins or an error message.
        """
        print(f"🎯 [DEBUG] get_coins_list CHAMADA! include_platform='{include_platform}'")

        try:
            # Parameters for the coins list endpoint - convert boolean to string
            params = {
                "include_platform": str(include_platform).lower()
            }

            response_data = self._make_request("coins/list", params)
            return self._format_coins_list_response(response_data)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching coins list: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting coins list: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_coins_list_response(self, coins_data: List[Dict]) -> str:
        """
        Format the coins list response into a readable string.
        """
        if not coins_data:
            no_data_msg = "No coins data found"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        total_coins = len(coins_data)
        
        # Show first 10 coins as examples
        example_coins = coins_data[:10]
        
        result = f"""
🪙 **CoinGecko Supported Cryptocurrencies**
📊 **Total Available**: {total_coins} cryptocurrencies

**Sample of Available Coins (First 10):**
        """.strip()
        
        for coin in example_coins:
            coin_id = coin.get('id', 'N/A')
            symbol = coin.get('symbol', '').upper()
            name = coin.get('name', 'N/A')
            
            result += f"\n• **{name} ({symbol})** - ID: `{coin_id}`"
        
        result += f"\n\n💡 **Note**: This is just a sample. Total {total_coins} coins are available. Use specific coin IDs with other functions."

        print(f"✅ [DEBUG] Lista de moedas formatada com sucesso - {total_coins} moedas encontradas")
        return result
