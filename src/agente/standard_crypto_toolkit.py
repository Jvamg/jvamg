"""
Standard Crypto Analysis ToolKit
===============================

ToolKit customizado para o framework Agno que combina funcionalidades 
dos toolkits CoinGecko e CoinDesk.
"""

from typing import Dict, List, Optional, Any
from agno.tools import Toolkit
from coingeckoToolKit import CoinGeckoToolKit
from coindeskToolKit import CoinDeskToolKit


class StandardCryptoAnalysisToolKit(Toolkit):
    """
    ToolKit que combina funcionalidades dos toolkits CoinGecko e CoinDesk
    """

    def __init__(self):
        super().__init__(name="standard_crypto_analysis")
        self.coingecko = CoinGeckoToolKit()
        self.coindesk = CoinDeskToolKit()

        # Registrar ferramentas básicas de análise
        self.register(self.get_comprehensive_market_analysis)
        self.register(self.get_crypto_overview)

    def get_comprehensive_market_analysis(
        self,
        coin_id: str,
        include_news: bool = True
    ) -> str:
        """
        Get comprehensive market analysis combining data from multiple sources.

        Args:
            coin_id (str): ID da criptomoeda no CoinGecko (ex: 'bitcoin', 'ethereum')
            include_news (bool): Se deve incluir análise de notícias

        Returns:
            str: Análise combinada dos dados
        """
        try:
            # Get basic coin data
            coin_data = self.coingecko.get_coin_data(coin_id)
            market_data = self.coingecko.get_market_data(coin_id)

            # Get technical analysis for different timeframes
            short_term_tech = self.coingecko.perform_technical_analysis(
                coin_id, "usd", "30")
            long_term_tech = self.coingecko.perform_technical_analysis(
                coin_id, "usd", "365")

            result = f"# 📊 Comprehensive Analysis for {coin_id.title()}\n\n"
            result += "## 💰 Market Data\n"
            result += market_data + "\n\n"

            result += "## 📈 Technical Analysis - Short Term (30 days)\n"
            result += short_term_tech + "\n\n"

            result += "## 📊 Technical Analysis - Long Term (365 days)\n"
            result += long_term_tech + "\n\n"

            # Add news if requested
            if include_news:
                try:
                    news = self.coindesk.get_latest_articles(
                        limit=10, category='CRYPTOCURRENCY')
                    result += "## 📰 Latest Market News\n"
                    result += news + "\n\n"
                except Exception as e:
                    result += f"## 📰 News Analysis\n⚠️ Could not fetch news: {str(e)}\n\n"

            result += "⚠️ *This analysis is for informational purposes only and does not constitute financial advice.*"

            return result

        except Exception as e:
            return f"❌ Error performing comprehensive analysis: {str(e)}"

    def get_crypto_overview(self, coin_id: str) -> str:
        """
        Get a quick overview of a cryptocurrency.

        Args:
            coin_id (str): ID da criptomoeda no CoinGecko

        Returns:
            str: Resumo rápido da criptomoeda
        """
        try:
            # Get essential data
            market_data = self.coingecko.get_market_data(coin_id)
            coin_data = self.coingecko.get_coin_data(coin_id)

            result = f"# 🚀 Quick Overview: {coin_id.title()}\n\n"
            result += "## 💰 Current Market Status\n"
            result += market_data + "\n\n"

            result += "## 📋 Coin Details\n"
            result += coin_data + "\n\n"

            result += "⚠️ *This is a basic overview. Use get_comprehensive_market_analysis for detailed analysis.*"

            return result

        except Exception as e:
            return f"❌ Error getting crypto overview: {str(e)}"
