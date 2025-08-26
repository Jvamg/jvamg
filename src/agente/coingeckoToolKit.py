import requests
import os
import pandas as pd
import pandas_ta as ta
from datetime import datetime
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
        self.default_vs_currency: str = os.getenv(
            "DEFAULT_VS_CURRENCY", "usd").lower()

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
        self.register(self.perform_technical_analysis)
        self.register(self.get_coin_symbol)

    def get_coin_symbol(self, coin_id: str) -> str:
        """
        Get the ticker symbol (uppercase) for a given CoinGecko coin_id.

        Args:
            coin_id (str): The CoinGecko ID (e.g., "bitcoin", "ethereum").

        Returns:
            str: Uppercase ticker symbol (e.g., BTC, ETH) or an error message.
        """
        print(f"🎯 [DEBUG] get_coin_symbol CHAMADA! coin_id='{coin_id}'")
        try:
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false",
            }
            response_data = self._make_request(f"coins/{coin_id}", params)
            symbol = None
            if isinstance(response_data, dict):
                symbol = response_data.get("symbol")
            if not symbol:
                print(f"❌ [DEBUG] Symbol not found for {coin_id}")
                return f"Symbol not found for {coin_id}"
            result = str(symbol).upper()
            print(f"✅ [DEBUG] Resolved symbol for {coin_id}: {result}")
            return result
        except Exception as e:
            error_msg = f"Error resolving symbol for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] {error_msg}")
            return error_msg

    def _currency_symbol(self, code: str) -> str:
        """
        Map a vs_currency code to a currency symbol/prefix.
        Returns empty string when unknown so we can fall back to appending the code.
        """
        code = (code or "").lower()
        return {
            "usd": "$",
            "eur": "€",
            "brl": "R$",
            "gbp": "£",
            "jpy": "¥",
            "cny": "¥",
            "aud": "A$",
            "cad": "C$",
            "chf": "CHF",
            "inr": "₹",
            "rub": "₽",
            "krw": "₩",
            "try": "₺",
            "mxn": "MX$",
            "zar": "R",
            "nzd": "NZ$",
            "btc": "₿",
            "eth": "Ξ",
        }.get(code, "")

    def _format_price_value(self, value: Any, vs_currency: str) -> str:
        """
        Format a price value with appropriate decimals and currency symbol.
        Uses 6 decimals when value < 0.01, otherwise 2.
        Falls back to appending the currency code when symbol is unknown.
        """
        if isinstance(value, (int, float)):
            symbol = self._currency_symbol(vs_currency)
            formatted = f"{value:,.2f}" if value >= 0.01 else f"{value:.6f}"
            return f"{symbol}{formatted}" if symbol else f"{formatted} {vs_currency.upper()}"
        return str(value)

    def _format_amount_value(self, value: Any, vs_currency: str) -> str:
        """
        Format a large monetary amount (e.g., market cap, volume) with 0 decimals
        and an appropriate currency symbol when available.
        """
        if isinstance(value, (int, float)):
            symbol = self._currency_symbol(vs_currency)
            formatted = f"{value:,.0f}"
            return f"{symbol}{formatted}" if symbol else f"{formatted} {vs_currency.upper()}"
        return str(value)

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
            headers = {"User-Agent": "AGNO-CoinGecko-Toolkit/1.0",
                       "Accept": "application/json"}
        else:
            # Construct direct CoinGecko Pro API endpoint URL
            if not self.api_key:
                raise ValueError(
                    "Missing COINGECKO_API_KEY environment variable. Please set your CoinGecko API key.")

            url = f"https://pro-api.coingecko.com/api/v3/{endpoint_path}"
            headers = {
                "x-cg-pro-api-key": self.api_key,
                "User-Agent": "AGNO-CoinGecko-Toolkit/1.0",
                "Accept": "application/json"
            }

        # Make the request
        response = requests.get(
            url, params=params, headers=headers, timeout=self.timeout)

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
        print(
            f"🎯 [DEBUG] get_market_data CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}'")

        try:
            if not vs_currency:
                vs_currency = self.default_vs_currency
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

        # Format numbers for better readability with proper currency handling
        price_formatted = self._format_price_value(price, vs_currency)
        market_cap_formatted = self._format_amount_value(
            market_cap, vs_currency)
        volume_formatted = self._format_amount_value(volume_24h, vs_currency)

        if isinstance(price_change_24h, (int, float)):
            change_formatted = f"{price_change_24h:+.2f}%"
        else:
            change_formatted = str(price_change_24h)

        result = f"""
🪙 **{name} ({symbol})**
💰 **Price**: {price_formatted}
📊 **24h Change**: {change_formatted}
🏦 **Market Cap**: {market_cap_formatted}
📈 **24h Volume**: {volume_formatted}
        """.strip()

        print(f"✅ [DEBUG] Dados formatados com sucesso para {coin_id}")
        return result

    def get_coin_data(self, coin_id: str, localization: bool = True, tickers: bool = True,
                      market_data: bool = True, community_data: bool = True,
                      developer_data: bool = True, sparkline: bool = False, vs_currency: str = "usd") -> str:
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
            if not vs_currency:
                vs_currency = self.default_vs_currency
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
            return self._format_coin_data_response(response_data, vs_currency)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching coin data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting coin data for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_coin_data_response(self, coin_data: Dict, vs_currency: str) -> str:
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
            price_map = market_info.get('current_price', {}) or {}
            cap_map = market_info.get('market_cap', {}) or {}
            # Prefer requested currency, fall back to USD when not available
            price_val = price_map.get(
                vs_currency.lower(), price_map.get('usd'))
            cap_val = cap_map.get(vs_currency.lower(), cap_map.get('usd'))

            if isinstance(price_val, (int, float)):
                current_price = self._format_price_value(
                    price_val, vs_currency)
            if isinstance(cap_val, (int, float)):
                market_cap = self._format_amount_value(cap_val, vs_currency)

        # Extract description
        description = ""
        if 'description' in coin_data and 'en' in coin_data['description']:
            desc_text = coin_data['description']['en']
            if desc_text:
                # Truncate description to first 200 characters
                description = desc_text[:200] + \
                    "..." if len(desc_text) > 200 else desc_text

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

        print(
            f"✅ [DEBUG] Dados completos formatados com sucesso para {coin_data.get('id', 'unknown')}")
        return result

    def get_coin_history(self, coin_id: str, date: str, localization: bool = False, vs_currency: str = "usd") -> str:
        """
        Get historical market data for a specific cryptocurrency on a given date.

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            date (str): Date in format DD-MM-YYYY (e.g., "15-05-2022").
            localization (bool): Include localized language fields. Default is False.

        Returns:
            str: A formatted string containing historical market data or an error message.
        """
        print(
            f"🎯 [DEBUG] get_coin_history CHAMADA! coin_id='{coin_id}', date='{date}'")

        try:
            if not vs_currency:
                vs_currency = self.default_vs_currency
            # Parameters for the historical data endpoint - convert boolean to string
            params = {
                "date": date,
                "localization": str(localization).lower()
            }

            response_data = self._make_request(
                f"coins/{coin_id}/history", params)
            return self._format_coin_history_response(response_data, date, vs_currency)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching historical data for {coin_id} on {date}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting historical data for {coin_id} on {date}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_coin_history_response(self, history_data: Dict, date: str, vs_currency: str) -> str:
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

            price_map = market_info.get('current_price', {}) or {}
            cap_map = market_info.get('market_cap', {}) or {}
            vol_map = market_info.get('total_volume', {}) or {}

            price_val = price_map.get(
                vs_currency.lower(), price_map.get('usd'))
            cap_val = cap_map.get(vs_currency.lower(), cap_map.get('usd'))
            vol_val = vol_map.get(vs_currency.lower(), vol_map.get('usd'))

            if isinstance(price_val, (int, float)):
                current_price = self._format_price_value(
                    price_val, vs_currency)
            if isinstance(cap_val, (int, float)):
                market_cap = self._format_amount_value(cap_val, vs_currency)
            if isinstance(vol_val, (int, float)):
                total_volume = self._format_amount_value(vol_val, vs_currency)

        result = f"""
📅 **Historical Data for {name} ({symbol}) - {date}**
💰 **Price**: {current_price}
🏦 **Market Cap**: {market_cap}
📈 **24h Volume**: {total_volume}
        """.strip()

        print(
            f"✅ [DEBUG] Dados históricos formatados com sucesso para {history_data.get('id', 'unknown')} em {date}")
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
        print(
            f"🎯 [DEBUG] get_coin_chart CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}', days='{days}'")

        try:
            if not vs_currency:
                vs_currency = self.default_vs_currency
            # Parameters for the chart data endpoint
            params = {
                "vs_currency": vs_currency,
                "days": days
            }

            if interval:
                params["interval"] = interval
            if precision:
                params["precision"] = precision

            response_data = self._make_request(
                f"coins/{coin_id}/market_chart", params)
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
            first_formatted = self._format_price_value(
                first_price, vs_currency)
            last_formatted = self._format_price_value(last_price, vs_currency)
            high_formatted = self._format_price_value(
                highest_price, vs_currency)
            low_formatted = self._format_price_value(lowest_price, vs_currency)
            change_formatted = f"{price_change:+.2f}%"

            result = f"""
📈 **Chart Data for {coin_id.title()} - Last {days} day(s)**
💰 **Period Start**: {first_formatted}
💰 **Period End**: {last_formatted}
📊 **Change**: {change_formatted}
📈 **Highest**: {high_formatted}
📉 **Lowest**: {low_formatted}
📅 **Data Points**: {len(prices)} price entries, {len(market_caps)} market cap entries, {len(total_volumes)} volume entries
            """.strip()
        else:
            result = f"No price data available for {coin_id} in the last {days} day(s)"

        print(
            f"✅ [DEBUG] Dados de gráfico formatados com sucesso para {coin_id}")
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
        print(
            f"🎯 [DEBUG] get_coin_ohlc CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}', days='{days}'")

        try:
            if not vs_currency:
                vs_currency = self.default_vs_currency
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
            open_formatted = self._format_price_value(open_price, vs_currency)
            high_formatted = self._format_price_value(high_price, vs_currency)
            low_formatted = self._format_price_value(low_price, vs_currency)
            close_formatted = self._format_price_value(
                close_price, vs_currency)

            # Calculate change percentage
            if open_price > 0:
                change_percent = (
                    (close_price - open_price) / open_price) * 100
                change_formatted = f"{change_percent:+.2f}%"
            else:
                change_formatted = "N/A"

            result = f"""
🕯️ **OHLC Data for {coin_id.title()} - Last {days} day(s)**
🟢 **Open**: {open_formatted}
📈 **High**: {high_formatted}
📉 **Low**: {low_formatted}
🔴 **Close**: {close_formatted}
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
        print(
            f"🎯 [DEBUG] get_coins_list CHAMADA! include_platform='{include_platform}'")

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

        print(
            f"✅ [DEBUG] Lista de moedas formatada com sucesso - {total_coins} moedas encontradas")
        return result

    def perform_technical_analysis(self, coin_id: str, vs_currency: str = "usd", days: str = "90") -> str:
        """
        Perform comprehensive technical analysis for a cryptocurrency with intelligent multi-timeframe support.

        This function fetches market chart data (close prices) and calculates key technical indicators:
        - RSI (Relative Strength Index): Only for short/medium-term analysis (≤90 days) - identifies overbought/oversold conditions
        - MACD (Moving Average Convergence Divergence): Identifies momentum changes  
        - Moving Averages (20, 50 days): Short and medium-term trends
        - Moving Average (200 days): Long-term trend (only for periods >= 200 days)

        Multi-Timeframe Logic:
        - For periods ≤ 90 days: Calculates RSI, MACD, SMA20/50 (short/medium-term focus)
        - For periods > 90 days: Skips RSI (not meaningful for long-term), focuses on MACD and SMAs
        - For periods ≥ 200 days: Includes SMA200 for complete long-term trend analysis

        Note: Uses market_chart endpoint with daily interval for consistent granularity

        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currency (str): The target currency for price data (e.g., "usd", "eur", "brl"). Default is "usd".
            days (str): Number of days of historical data to analyze. Default is "90".

        Returns:
            str: A formatted string containing technical analysis conclusions or an error message.
        """
        print(
            f"🎯 [DEBUG] perform_technical_analysis CHAMADA! coin_id='{coin_id}', vs_currency='{vs_currency}', days='{days}'")

        try:
            # Smart multi-timeframe logic: allow shorter periods but adjust indicators accordingly
            requested_days = int(days) if days.isdigit() else 90

            if requested_days < 200:
                # For short-term analysis: ensure minimum bars for MACD (>=35), SMA 200 will be unavailable
                min_macd_days = 35
                actual_days_int = max(requested_days, min_macd_days)
                actual_days = str(actual_days_int)
                if actual_days_int == requested_days:
                    print(
                        f"🔍 [DEBUG] Using {actual_days} days for short-term analysis (SMA 200 may not be available)")
                else:
                    print(
                        f"🔍 [DEBUG] Adjusting days from {requested_days} to {actual_days} to satisfy MACD minimum (35) for short-term analysis")
            else:
                # For long-term analysis: ensure minimum 200 days for all indicators
                actual_days = str(max(200, requested_days))
                if actual_days != days:
                    print(
                        f"🔍 [DEBUG] Adjusting days from {days} to {actual_days} to ensure sufficient data for all indicators")

            # Get market chart data from the API (provides more data points than OHLC)
            # Force daily interval for consistent granularity across all timeframes
            print(f"🔍 [DEBUG] Making API request for market chart data...")
            response_data = self._make_request(f"coins/{coin_id}/market_chart", {
                "vs_currency": vs_currency,
                "days": actual_days,
                "interval": "daily"  # Force daily data for consistent RSI calculations
            })

            print(f"🔍 [DEBUG] API response received")
            print(f"🔍 [DEBUG] Response data type: {type(response_data)}")
            print(f"🔍 [DEBUG] Response data is None: {response_data is None}")

            if response_data is not None:
                prices = response_data.get('prices', [])
                print(f"🔍 [DEBUG] Prices data length: {len(prices)}")
                print(
                    f"🔍 [DEBUG] First few price entries: {prices[:3] if len(prices) >= 3 else prices}")

                # Extract close prices from market_chart format: [[timestamp, close_price], ...]
                # Filter out entries with missing/non-numeric prices to avoid indicator errors
                close_data = [
                    [ts, px] for ts, px in prices if isinstance(px, (int, float))
                ]
                if len(close_data) != len(prices):
                    print(
                        f"⚠️ [DEBUG] Filtered out {len(prices) - len(close_data)} non-numeric/None price entries")
            else:
                close_data = []

            if not close_data or len(close_data) == 0:
                print(f"❌ [DEBUG] No market chart data received from API")
                return f"❌ Insufficient market data for technical analysis of {coin_id}"

            if len(close_data) < 200:
                print(
                    f"⚠️ [DEBUG] Warning: Only {len(close_data)} data points available, may affect SMA 200 calculation")

            print(
                f"🔍 [DEBUG] Proceeding to technical calculations with {len(close_data)} data points")
            return self._perform_technical_calculations(close_data, coin_id, vs_currency, actual_days, requested_days)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching data for technical analysis of {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error during technical analysis of {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _perform_technical_calculations(self, close_data: List, coin_id: str, vs_currency: str, days: str, requested_days: int) -> str:
        """
        Perform the actual technical analysis calculations using pandas and pandas-ta.
        Uses only close prices from market_chart data format: [[timestamp, close_price], ...]

        Args:
            close_data: List of [timestamp, close_price] pairs from market_chart API
            coin_id: Cryptocurrency identifier  
            vs_currency: Target currency (e.g., 'usd')
            days: Actual days of data fetched from API
            requested_days: Original days requested by user (affects which indicators to calculate)
        """
        try:
            print(f"🔍 [DEBUG] Starting technical calculations for {coin_id}")
            print(f"🔍 [DEBUG] Raw close data length: {len(close_data)}")
            print(
                f"🔍 [DEBUG] First 3 close entries: {close_data[:3] if len(close_data) >= 3 else close_data}")

            # Convert close data to DataFrame: [[timestamp, close_price], ...] -> DataFrame
            df = pd.DataFrame(close_data, columns=['timestamp', 'close'])
            print(f"🔍 [DEBUG] DataFrame created with shape: {df.shape}")
            print(f"🔍 [DEBUG] DataFrame columns: {df.columns.tolist()}")
            print(f"🔍 [DEBUG] DataFrame head:\n{df.head()}")

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            # Drop any remaining NaNs in 'close' to ensure stable indicator calculations
            before_drop = len(df)
            df = df[df['close'].notna()]
            after_drop = len(df)
            if after_drop != before_drop:
                print(
                    f"⚠️ [DEBUG] Dropped {before_drop - after_drop} rows with NaN close values before indicators")

            print(f"🔍 [DEBUG] DataFrame after processing - shape: {df.shape}")
            print(f"🔍 [DEBUG] DataFrame dtypes:\n{df.dtypes}")
            print(f"🔍 [DEBUG] DataFrame statistics:\n{df.describe()}")
            print(f"🔍 [DEBUG] Any NaN values: {df.isnull().sum().sum()}")

            # Ensure we have enough data points
            if len(df) < 20:
                print(
                    f"❌ [DEBUG] Insufficient data: {len(df)} points, need at least 20")
                return f"❌ Insufficient data points ({len(df)}) for reliable technical analysis of {coin_id}. Need at least 20 data points."

            # Calculate technical indicators with error handling
            # RSI (14-period) - Only for short to medium-term analysis
            current_rsi = None
            rsi_interpretation = ""

            if requested_days <= 90:
                try:
                    print(
                        f"🔍 [DEBUG] Starting RSI calculation for {coin_id} (short/medium-term analysis)")
                    print(
                        f"🔍 [DEBUG] Close prices sample: {df['close'].head()}")
                    print(f"🔍 [DEBUG] Close prices length: {len(df['close'])}")

                    rsi_series = ta.rsi(df['close'], length=14)
                    print(f"🔍 [DEBUG] RSI series type: {type(rsi_series)}")
                    print(
                        f"🔍 [DEBUG] RSI series is None: {rsi_series is None}")
                    if rsi_series is not None:
                        print(
                            f"🔍 [DEBUG] RSI series is empty: {rsi_series.empty}")
                        print(
                            f"🔍 [DEBUG] RSI series length: {len(rsi_series)}")
                        print(
                            f"🔍 [DEBUG] RSI series last 5 values:\n{rsi_series.tail()}")

                    if rsi_series is None or rsi_series.empty:
                        print(
                            f"❌ [DEBUG] RSI calculation failed - series is None or empty")
                        return f"❌ Failed to calculate RSI for {coin_id}. Data may be insufficient or invalid."

                    df['rsi'] = rsi_series
                    current_rsi = df['rsi'].iloc[-1]
                    print(f"🔍 [DEBUG] Current RSI value: {current_rsi}")

                    if pd.isna(current_rsi):
                        print(f"❌ [DEBUG] Current RSI is NaN")
                        return f"❌ RSI calculation returned NaN for {coin_id}. Data may be insufficient."

                    print(
                        f"✅ [DEBUG] RSI calculation successful: {current_rsi}")
                except Exception as e:
                    print(f"❌ [DEBUG] RSI calculation exception: {str(e)}")
                    return f"❌ Error calculating RSI for {coin_id}: {str(e)}"
            else:
                print(
                    f"🔍 [DEBUG] RSI skipped - not relevant for long-term analysis ({requested_days} days)")
                current_rsi = None

            # MACD (12, 26, 9)
            try:
                print(f"🔍 [DEBUG] Starting MACD calculation for {coin_id}")
                # Ensure numeric close series without None/NaN values
                close_series = pd.to_numeric(
                    df['close'], errors='coerce').dropna()
                print(
                    f"🔍 [DEBUG] Close series length for MACD: {len(close_series)}")
                # Require at least slow(26) + signal(9) bars to produce a valid last value
                if len(close_series) < 35:
                    print(
                        f"❌ [DEBUG] Not enough data for MACD: have {len(close_series)}, need >= 35")
                    return f"❌ Insufficient data points ({len(close_series)}) for MACD calculation for {coin_id}. Need at least 35."

                # Primary: use pandas_ta.macd; Fallback: manual EMA-based MACD if it fails
                try:
                    macd_data = ta.macd(close_series)
                except Exception as e_macd:
                    print(
                        f"⚠️ [DEBUG] pandas_ta.macd failed, falling back to manual MACD: {e_macd}")
                    ema_fast = close_series.ewm(
                        span=12, adjust=False, min_periods=12).mean()
                    ema_slow = close_series.ewm(
                        span=26, adjust=False, min_periods=26).mean()
                    macd_line = ema_fast - ema_slow
                    signal_line = macd_line.ewm(
                        span=9, adjust=False, min_periods=9).mean()
                    hist_line = macd_line - signal_line
                    macd_data = pd.DataFrame({
                        'MACD_12_26_9': macd_line,
                        'MACDs_12_26_9': signal_line,
                        'MACDh_12_26_9': hist_line,
                    })

                print(f"🔍 [DEBUG] MACD data type: {type(macd_data)}")
                print(f"🔍 [DEBUG] MACD data is None: {macd_data is None}")

                if macd_data is not None:
                    print(f"🔍 [DEBUG] MACD data is empty: {macd_data.empty}")
                    print(
                        f"🔍 [DEBUG] MACD data columns: {getattr(macd_data, 'columns', [])}")
                    print(
                        f"🔍 [DEBUG] MACD data shape: {getattr(macd_data, 'shape', None)}")

                if macd_data is None or (hasattr(macd_data, 'empty') and macd_data.empty):
                    print(
                        f"❌ [DEBUG] MACD calculation failed - data is None or empty")
                    return f"❌ Failed to calculate MACD for {coin_id}. Data may be insufficient or invalid."

                # Check if required columns exist
                expected_columns = ['MACD_12_26_9',
                                    'MACDs_12_26_9', 'MACDh_12_26_9']
                missing_columns = [
                    col for col in expected_columns if col not in macd_data.columns]
                if missing_columns:
                    print(f"❌ [DEBUG] Missing MACD columns: {missing_columns}")
                    print(
                        f"🔍 [DEBUG] Available columns: {macd_data.columns.tolist()}")
                    return f"❌ MACD calculation did not return expected columns for {coin_id}. Missing: {missing_columns}"

                # Coerce any non-numeric/None to NaN and drop incomplete rows for current value checks
                for col in expected_columns:
                    macd_data[col] = pd.to_numeric(
                        macd_data[col], errors='coerce')
                macd_valid = macd_data[expected_columns].dropna()
                print(
                    f"🔍 [DEBUG] MACD valid rows after dropna: {len(macd_valid)}")
                if macd_valid.empty:
                    print(
                        f"❌ [DEBUG] MACD calculation resulted in all-NaN values")
                    return f"❌ MACD calculation returned NaN values for {coin_id}. Data may be insufficient."

                # Align with df index and also keep for reading last non-null values
                df['macd'] = macd_data['MACD_12_26_9']
                df['macd_signal'] = macd_data['MACDs_12_26_9']
                df['macd_histogram'] = macd_data['MACDh_12_26_9']

                last_row = macd_valid.iloc[-1]
                current_macd = float(last_row['MACD_12_26_9'])
                current_signal = float(last_row['MACDs_12_26_9'])
                current_histogram = float(last_row['MACDh_12_26_9'])

                print(
                    f"🔍 [DEBUG] Current MACD values - MACD: {current_macd}, Signal: {current_signal}, Histogram: {current_histogram}")
                print(f"✅ [DEBUG] MACD calculation successful")
            except Exception as e:
                print(f"❌ [DEBUG] MACD calculation exception: {str(e)}")
                return f"❌ Error calculating MACD for {coin_id}: {str(e)}"

            # Check for MACD crossover (bullish/bearish signals)
            macd_crossover = "Neutro"
            # Use valid histogram rows only for crossover detection
            try:
                macd_hist_valid = pd.to_numeric(
                    df['macd_histogram'], errors='coerce').dropna()
                if len(macd_hist_valid) >= 2:
                    prev_histogram = float(macd_hist_valid.iloc[-2])
                    if current_histogram > 0 and prev_histogram <= 0:
                        macd_crossover = "Cruzamento de Alta (Bullish)"
                    elif current_histogram < 0 and prev_histogram >= 0:
                        macd_crossover = "Cruzamento de Baixa (Bearish)"
            except Exception as e:
                print(
                    f"⚠️ [DEBUG] MACD crossover detection skipped due to error: {e}")

            # Moving Averages with error handling
            try:
                print(
                    f"🔍 [DEBUG] Starting Moving Averages calculation for {coin_id}")
                current_price = df['close'].iloc[-1]
                print(f"🔍 [DEBUG] Current price: {current_price}")

                # SMA 20
                print(f"🔍 [DEBUG] Calculating SMA 20")
                sma_20_series = ta.sma(df['close'], length=20)
                print(
                    f"🔍 [DEBUG] SMA 20 series type: {type(sma_20_series)}, is None: {sma_20_series is None}")
                if sma_20_series is not None:
                    print(
                        f"🔍 [DEBUG] SMA 20 series empty: {sma_20_series.empty}, length: {len(sma_20_series)}")

                if sma_20_series is None or sma_20_series.empty:
                    print(f"❌ [DEBUG] SMA 20 calculation failed")
                    return f"❌ Failed to calculate SMA 20 for {coin_id}. Data may be insufficient or invalid."

                df['sma_20'] = sma_20_series
                sma_20 = df['sma_20'].iloc[-1]
                print(f"🔍 [DEBUG] SMA 20 value: {sma_20}")
                if pd.isna(sma_20):
                    print(f"❌ [DEBUG] SMA 20 is NaN")
                    return f"❌ SMA 20 calculation returned NaN for {coin_id}. Data may be insufficient."

                # SMA 50
                print(f"🔍 [DEBUG] Calculating SMA 50")
                sma_50_series = ta.sma(df['close'], length=50)
                print(
                    f"🔍 [DEBUG] SMA 50 series type: {type(sma_50_series)}, is None: {sma_50_series is None}")
                if sma_50_series is not None:
                    print(
                        f"🔍 [DEBUG] SMA 50 series empty: {sma_50_series.empty}, length: {len(sma_50_series)}")

                if sma_50_series is None or sma_50_series.empty:
                    print(f"❌ [DEBUG] SMA 50 calculation failed")
                    return f"❌ Failed to calculate SMA 50 for {coin_id}. Data may be insufficient or invalid."

                df['sma_50'] = sma_50_series
                sma_50 = df['sma_50'].iloc[-1]
                print(f"🔍 [DEBUG] SMA 50 value: {sma_50}")
                if pd.isna(sma_50):
                    print(f"❌ [DEBUG] SMA 50 is NaN")
                    return f"❌ SMA 50 calculation returned NaN for {coin_id}. Data may be insufficient."

                # SMA 200 (only if we have enough data AND it's appropriate for the requested timeframe)
                sma_200 = None
                print(
                    f"🔍 [DEBUG] Checking SMA 200 eligibility - have {len(df)} data points, requested {requested_days} days")
                if len(df) >= 200 and requested_days >= 200:
                    print(f"🔍 [DEBUG] Calculating SMA 200")
                    sma_200_series = ta.sma(df['close'], length=200)
                    print(
                        f"🔍 [DEBUG] SMA 200 series type: {type(sma_200_series)}, is None: {sma_200_series is None}")
                    if sma_200_series is not None:
                        print(
                            f"🔍 [DEBUG] SMA 200 series empty: {sma_200_series.empty}, length: {len(sma_200_series)}")

                    if sma_200_series is not None and not sma_200_series.empty:
                        df['sma_200'] = sma_200_series
                        sma_200_value = df['sma_200'].iloc[-1]
                        print(f"🔍 [DEBUG] SMA 200 value: {sma_200_value}")
                        if not pd.isna(sma_200_value):
                            sma_200 = sma_200_value
                            print(
                                f"✅ [DEBUG] SMA 200 calculated successfully: {sma_200}")
                        else:
                            print(f"⚠️ [DEBUG] SMA 200 is NaN")
                    else:
                        print(
                            f"⚠️ [DEBUG] SMA 200 calculation failed or returned empty")
                elif len(df) < 200:
                    print(
                        f"⚠️ [DEBUG] Not enough data for SMA 200 ({len(df)} points)")
                else:
                    print(
                        f"⚠️ [DEBUG] SMA 200 not calculated - short-term analysis ({requested_days} days requested)")

                print(f"✅ [DEBUG] Moving Averages calculation completed")

            except Exception as e:
                print(
                    f"❌ [DEBUG] Moving Averages calculation exception: {str(e)}")
                return f"❌ Error calculating Moving Averages for {coin_id}: {str(e)}"

            # Trend Analysis
            price_vs_sma20 = "acima" if current_price > sma_20 else "abaixo"
            price_vs_sma50 = "acima" if current_price > sma_50 else "abaixo"

            # Golden Cross / Death Cross analysis
            cross_analysis = ""
            if sma_200 is not None:
                price_vs_sma200 = "acima" if current_price > sma_200 else "abaixo"
                if sma_50 > sma_200:
                    cross_analysis = "📈 Configuração de Alta (SMA50 > SMA200)"
                else:
                    cross_analysis = "📉 Configuração de Baixa (SMA50 < SMA200)"
            else:
                if requested_days < 200:
                    price_vs_sma200 = "N/A (análise de curto prazo)"
                else:
                    price_vs_sma200 = "N/A (dados insuficientes)"

            # RSI interpretation
            if current_rsi is not None:
                if current_rsi > 70:
                    rsi_interpretation = "🔴 Sobrecompra (possível correção)"
                elif current_rsi < 30:
                    rsi_interpretation = "🟢 Sobrevenda (possível recuperação)"
                else:
                    rsi_interpretation = "🟡 Neutro"
            else:
                rsi_interpretation = "N/A (análise de longo prazo)"

            # Overall trend assessment
            bullish_signals = 0
            bearish_signals = 0

            # RSI signals (only for short/medium-term analysis)
            if current_rsi is not None:
                if current_rsi < 30:
                    bullish_signals += 1
                elif current_rsi > 70:
                    bearish_signals += 1

            # Price vs moving averages
            if current_price > sma_20:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if current_price > sma_50:
                bullish_signals += 1
            else:
                bearish_signals += 1

            # MACD signals
            if current_macd > current_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1

            # Overall sentiment
            if bullish_signals > bearish_signals:
                overall_sentiment = "🟢 Tendência de Alta"
            elif bearish_signals > bullish_signals:
                overall_sentiment = "🔴 Tendência de Baixa"
            else:
                overall_sentiment = "🟡 Tendência Neutra/Lateral"

            # Format the result
            print(f"🔍 [DEBUG] Formatting final result for {coin_id}")
            print(
                f"🔍 [DEBUG] Final values - RSI: {current_rsi}, MACD: {current_macd}, SMA20: {sma_20}, SMA50: {sma_50}, SMA200: {sma_200}")
            print(
                f"🔍 [DEBUG] Sentiment analysis - Bullish: {bullish_signals}, Bearish: {bearish_signals}, Overall: {overall_sentiment}")

            # Format RSI section based on availability
            rsi_section = ""
            if current_rsi is not None:
                rsi_section = f"📈 **RSI (14)**: {current_rsi:.2f} - {rsi_interpretation}"
            else:
                rsi_section = f"📈 **RSI**: {rsi_interpretation}"

            current_price_formatted = self._format_price_value(
                current_price, vs_currency)
            sma_20_formatted = self._format_price_value(sma_20, vs_currency)
            sma_50_formatted = self._format_price_value(sma_50, vs_currency)
            sma_200_display = f"{self._format_price_value(sma_200, vs_currency)} (preço está {price_vs_sma200})" if sma_200 is not None else f"{price_vs_sma200}"

            result = f"""
📊 **Análise Técnica - {coin_id.upper()}** ({days} dias)
💰 **Preço Atual**: {current_price_formatted}

🔍 **Indicadores Técnicos:**

{rsi_section}

📊 **MACD**:
   • MACD: {current_macd:.6f}
   • Signal: {current_signal:.6f}
   • Histograma: {current_histogram:.6f}
   • Status: {macd_crossover}

📏 **Médias Móveis**:
   • SMA 20: {sma_20_formatted} (preço está {price_vs_sma20})
   • SMA 50: {sma_50_formatted} (preço está {price_vs_sma50})
   • SMA 200: {sma_200_display}
   • {cross_analysis if cross_analysis else 'Análise de cruzamento indisponível'}

🎯 **Resumo da Análise**:
{overall_sentiment}

📋 **Sinais**: {bullish_signals} sinais de alta vs {bearish_signals} sinais de baixa

⚠️ **Aviso**: Esta é uma análise técnica baseada em dados históricos e não constitui aconselhamento financeiro.
            """.strip()

            print(
                f"✅ [DEBUG] Análise técnica realizada com sucesso para {coin_id}")
            print(f"🔍 [DEBUG] Result length: {len(result)} characters")
            return result

        except Exception as e:
            error_msg = f"Error during technical calculations for {coin_id}: {str(e)}"
            print(f"❌ [DEBUG] Erro nos cálculos técnicos: {error_msg}")
            return error_msg
