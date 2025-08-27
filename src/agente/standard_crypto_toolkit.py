"""
Standard Crypto Analysis ToolKit
===============================

ToolKit customizado para o framework Agno que gera análises de criptomoedas
com output padronizado usando o sistema OutputFormatter.
"""

from typing import Dict, List, Optional, Any
from agno.tools import Toolkit
from coingeckoToolKit import CoinGeckoToolKit
from coindeskToolKit import CoinDeskToolKit
from agno_output_adapter import AgnoOutputAdapter
from output_formatter import OutputFormatter, AnalysisType


class StandardCryptoAnalysisToolKit(Toolkit):
    """
    ToolKit que combina funcionalidades existentes com output padronizado
    """

    def __init__(self):
        super().__init__(name="standard_crypto_analysis")
        self.coingecko = CoinGeckoToolKit()
        self.coindesk = CoinDeskToolKit()
        self.adapter = AgnoOutputAdapter()
        self.formatter = OutputFormatter()

        # Registrar ferramentas
        self.register(self.comprehensive_crypto_analysis)
        self.register(self.quick_crypto_summary)
        self.register(self.multi_crypto_comparison)

    def _create_raw_data_section(self, coin_data: str, market_data: str, 
                                short_tech: str, long_tech: str, 
                                news_data: str, news_count: int) -> str:
        """
        Cria seção com dados RAW PUROS dos endpoints (sem formatação)
        """
        return f"""
### 🔍 **Market Data Endpoint RAW**
```json
{market_data}
```

### 📊 **Technical Analysis RAW - Short Term (30d)**
```
{short_tech}
```

### 📈 **Technical Analysis RAW - Long Term (365d)** 
```
{long_tech}
```

### 📰 **News Data RAW** ({news_count} articles)
```json
{news_data if news_data else "No news data available"}
```

### 💡 **Validation Instructions for Agent**
Use Reasoning/Thinking Tools to analyze the RAW data above and verify:
- RSI values are in valid range (0-100) and make sense
- MACD line/signal relationships support trend conclusions
- SMA values have logical ordering and reasonable prices
- Market data consistency (price, volume, market cap correlation)
- News sentiment aligns with technical indicators
- Short-term vs long-term trend consistency
"""



    def comprehensive_crypto_analysis(
        self,
        coin_id: str,
        include_sentiment: bool = True,
        output_format: str = "markdown"
    ) -> str:
        """
        Realiza análise abrangente de uma criptomoeda com output padronizado.

        Args:
            coin_id (str): ID da criptomoeda no CoinGecko (ex: 'bitcoin', 'ethereum')
            include_sentiment (bool): Se deve incluir análise de sentimento
            output_format (str): Formato do output ('markdown', 'json', 'summary')

        Returns:
            str: Análise formatada conforme especificado
        """
        try:
            # 1. Obter dados básicos da moeda
            coin_data_response = self.coingecko.get_coin_data(coin_id)
            market_data_response = self.coingecko.get_market_data(coin_id)

            # 2. Realizar análises técnicas multi-timeframe
            short_term_analysis = self.coingecko.perform_technical_analysis(
                coin_id, "usd", "30")
            long_term_analysis = self.coingecko.perform_technical_analysis(
                coin_id, "usd", "365")

            # 3. Análise de sentimento (opcional)
            news_response = ""
            news_count = 0
            if include_sentiment:
                try:
                    # Tentar obter símbolo da moeda com fallback
                    coin_symbol = self._extract_coin_symbol(
                        coin_data_response, coin_id)

                    print(
                        f"📰 [DEBUG] Buscando notícias específicas para: '{coin_symbol}'")
                    specific_news = self.coindesk.get_latest_articles(
                        limit=10, category=coin_symbol)
                    specific_count = self._count_articles_in_response(
                        specific_news)

                    print(
                        f"📰 [DEBUG] Buscando notícias gerais de mercado: 'CRYPTOCURRENCY'")
                    market_news = self.coindesk.get_latest_articles(
                        limit=10, category='CRYPTOCURRENCY')
                    market_count = self._count_articles_in_response(
                        market_news)

                    # Combinar as duas respostas
                    news_response = f"""=== NOTÍCIAS ESPECÍFICAS DE {coin_symbol} ===
{specific_news}

=== NOTÍCIAS GERAIS DO MERCADO ===
{market_news}"""

                    news_count = specific_count + market_count
                    print(
                        f"📊 [DEBUG] Total de artigos coletados: {news_count} ({specific_count} específicos + {market_count} de mercado)")

                except Exception as e:
                    print(f"Warning: Could not fetch sentiment data: {e}")
                    # Fallback: tentar só notícias gerais se específicas falharem
                    try:
                        print(
                            f"📰 [DEBUG] Fallback: buscando apenas notícias gerais de mercado")
                        news_response = self.coindesk.get_latest_articles(
                            limit=15, category='CRYPTOCURRENCY')
                        news_count = self._count_articles_in_response(
                            news_response)
                    except:
                        pass

            # 4. Extrair nome e símbolo
            coin_name, coin_symbol = self._extract_coin_info(
                coin_data_response)

            # 5. Criar análise estruturada
            analysis = self.adapter.create_comprehensive_analysis(
                coin_id=coin_id,
                coin_name=coin_name,
                coin_symbol=coin_symbol,
                market_response=market_data_response,
                short_term_technical=short_term_analysis,
                long_term_technical=long_term_analysis,
                news_response=news_response,
                news_count=news_count
            )

            # 6. Formatar output e incluir dados RAW para validação
            if output_format.lower() == "json":
                formatted_result = self.formatter.format_json_output(analysis)
            elif output_format.lower() == "summary":
                formatted_result = self.formatter.format_compact_summary(analysis)
            else:  # default to markdown
                formatted_result = self.formatter.format_markdown_output(analysis)
            
            # 6.1. Anexar dados RAW dos endpoints para validação posterior (sem gastar APIs)
            raw_data_section = self._create_raw_data_section(
                coin_data_response, market_data_response, 
                short_term_analysis, long_term_analysis, 
                news_response, news_count
            )
            
            # 6.2. Combinar output padronizado + dados RAW
            final_output = f"""{formatted_result}

---

## 📋 **Dados RAW dos Endpoints** (Para Validação)

{raw_data_section}

**⚠️ INSTRUÇÃO PARA VALIDAÇÃO**: Use Reasoning Tools ou Thinking Tools para analisar se os dados acima fazem sentido:
- Verificar consistência entre indicadores técnicos
- Validar se valores estão em ranges realistas (RSI 0-100, preços positivos, etc.)
- Analisar se tendências de curto e longo prazo são coerentes
- Identificar possíveis inconsistências nos dados de mercado
- Pensar criticamente sobre a qualidade e confiabilidade da análise
"""
            
            return final_output

        except Exception as e:
            return f"❌ Erro ao realizar análise abrangente: {str(e)}\nPor favor, verifique se o coin_id '{coin_id}' é válido."

    def quick_crypto_summary(self, coin_id: str) -> str:
        """
        Gera resumo rápido de uma criptomoeda com informações essenciais.

        Args:
            coin_id (str): ID da criptomoeda no CoinGecko

        Returns:
            str: Resumo compacto formatado
        """
        try:
            # Obter dados essenciais
            market_data = self.coingecko.get_market_data(coin_id)
            coin_data = self.coingecko.get_coin_data(coin_id)

            # Análise técnica rápida (apenas curto prazo)
            quick_analysis = self.coingecko.perform_technical_analysis(
                coin_id, "usd", "30")

            # Extrair informações básicas
            coin_name, coin_symbol = self._extract_coin_info(coin_data)
            market_info = self.adapter.parse_market_data_response(market_data)
            tech_info = self.adapter.parse_agno_technical_analysis(
                quick_analysis)

            # Criar análise mínima
            analysis = self.adapter.create_timeframe_analysis(
                self.adapter.TimeframeType.SHORT_TERM,
                30,
                tech_info
            )

            # Formatar resumo compacto
            price = market_info.get("current_price", 0)
            change_pct = market_info.get("price_change_percentage_24h", 0)

            price_emoji = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "🟡"
            trend_emoji = self.formatter._get_trend_emoji(
                analysis.trend_direction)

            summary = f"""🚀 **{coin_name} ({coin_symbol.upper()})**

💰 **Preço:** ${price:,.6f if price < 1 else ,.2f} ({price_emoji} {change_pct:+.2f}%)
📈 **Tendência:** {trend_emoji} {analysis.trend_direction.value.title()}
📊 **Confiança:** {analysis.confidence_level:.0f}%

**🔍 Análise Técnica:**"""

            if analysis.technical_indicators.rsi:
                rsi_status = analysis.technical_indicators.rsi_status or "neutral"
                rsi_emoji = {"oversold": "🟢", "overbought": "🔴",
                             "neutral": "🟡"}.get(rsi_status, "🟡")
                summary += f"\n• RSI: {analysis.technical_indicators.rsi:.1f} {rsi_emoji} ({rsi_status})"

            if analysis.technical_indicators.macd_trend:
                macd_emoji = "🟢" if "bullish" in analysis.technical_indicators.macd_trend else "🔴"
                summary += f"\n• MACD: {analysis.technical_indicators.macd_trend.replace('_', ' ')} {macd_emoji}"

            summary += f"\n\n🎯 **Trading:** {analysis.trading_implications}"
            summary += "\n\n⚠️ *Esta é uma análise técnica, não um conselho financeiro*"

            return summary

        except Exception as e:
            return f"❌ Erro ao gerar resumo: {str(e)}"

    def multi_crypto_comparison(
        self,
        coin_ids: List[str],
        comparison_metric: str = "performance"
    ) -> str:
        """
        Compara múltiplas criptomoedas com output padronizado.

        Args:
            coin_ids (List[str]): Lista de IDs das criptomoedas
            comparison_metric (str): Métrica de comparação ('performance', 'technical', 'market_cap')

        Returns:
            str: Comparação formatada
        """
        if len(coin_ids) < 2:
            return "❌ É necessário fornecer pelo menos 2 criptomoedas para comparação."

        if len(coin_ids) > 5:
            coin_ids = coin_ids[:5]  # Limitar a 5 para evitar sobrecarga

        try:
            comparisons = []

            for coin_id in coin_ids:
                try:
                    # Obter dados básicos
                    market_data = self.coingecko.get_market_data(coin_id)
                    coin_data = self.coingecko.get_coin_data(coin_id)

                    # Análise técnica básica
                    if comparison_metric == "technical":
                        tech_data = self.coingecko.perform_technical_analysis(
                            coin_id, "usd", "30")
                        tech_info = self.adapter.parse_agno_technical_analysis(
                            tech_data)
                    else:
                        tech_info = {}

                    # Parsear dados
                    market_info = self.adapter.parse_market_data_response(
                        market_data)
                    coin_name, coin_symbol = self._extract_coin_info(coin_data)

                    comparisons.append({
                        "coin_id": coin_id,
                        "name": coin_name,
                        "symbol": coin_symbol,
                        "market_data": market_info,
                        "technical_data": tech_info
                    })

                except Exception as e:
                    print(f"Warning: Could not analyze {coin_id}: {e}")
                    continue

            if not comparisons:
                return "❌ Não foi possível analisar nenhuma das criptomoedas fornecidas."

            # Formatar comparação
            return self._format_comparison(comparisons, comparison_metric)

        except Exception as e:
            return f"❌ Erro na comparação: {str(e)}"

    def _extract_coin_info(self, coin_data_response: str) -> tuple:
        """Extrai nome e símbolo da moeda da resposta"""
        try:
            # Patterns para extrair nome e símbolo
            import re

            # Tentar extrair nome
            name_match = re.search(r'"name":\s*"([^"]+)"', coin_data_response)
            if not name_match:
                name_match = re.search(
                    r'name.*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', coin_data_response)
            name = name_match.group(1) if name_match else "Unknown"

            # Tentar extrair símbolo
            symbol_match = re.search(
                r'"symbol":\s*"([^"]+)"', coin_data_response)
            if not symbol_match:
                symbol_match = re.search(
                    r'symbol.*?([A-Z]{2,5})', coin_data_response)
            symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"

            return name, symbol

        except:
            return "Unknown", "UNKNOWN"

    def _extract_coin_symbol(self, coin_data_response: str, coin_id: str = None) -> str:
        """Extrai apenas o símbolo da moeda com fallback baseado em coin_id"""
        _, symbol = self._extract_coin_info(coin_data_response)
        print(
            f"🔍 [DEBUG] Símbolo extraído da API: '{symbol}' para coin_id='{coin_id}'")

        # Se não conseguiu extrair e temos coin_id, usar mapeamento conhecido
        if symbol == "UNKNOWN" and coin_id:
            symbol_mapping = {
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
                'binancecoin': 'BNB',
                'ripple': 'XRP',
                'cardano': 'ADA',
                'solana': 'SOL',
                'polkadot': 'DOT',
                'dogecoin': 'DOGE',
                'avalanche-2': 'AVAX',
                'polygon': 'MATIC',
                'chainlink': 'LINK',
                'litecoin': 'LTC',
                'bitcoin-cash': 'BCH',
                'stellar': 'XLM',
                'vechain': 'VET',
                'ethereum-classic': 'ETC',
                'filecoin': 'FIL',
                'tron': 'TRX',
                'monero': 'XMR',
                'eos': 'EOS'
            }
            fallback_symbol = symbol_mapping.get(coin_id.lower(), symbol)
            print(
                f"💡 [DEBUG] Usando fallback: '{coin_id}' → '{fallback_symbol}' (era '{symbol}')")
            symbol = fallback_symbol
        else:
            print(f"✅ [DEBUG] Símbolo válido extraído da API: '{symbol}'")

        print(f"🎯 [DEBUG] Símbolo final para busca de notícias: '{symbol}'")
        return symbol

    def _count_articles_in_response(self, news_response: str) -> int:
        """Conta número de artigos na resposta de notícias"""
        try:
            # Contar baseado em padrões comuns
            import re

            # Tentar contar por separadores comuns
            article_markers = [
                r'📰',
                r'Title:',
                r'Título:',
                r'\d+\.',  # Numeração
                r'---'     # Separadores
            ]

            max_count = 0
            for marker in article_markers:
                count = len(re.findall(marker, news_response, re.IGNORECASE))
                max_count = max(max_count, count)

            return min(max_count, 20)  # Limitar a 20

        except:
            return 0

    def _format_comparison(self, comparisons: List[Dict], metric: str) -> str:
        """Formata comparação entre criptomoedas"""

        if metric == "performance":
            # Ordenar por performance 24h
            comparisons.sort(key=lambda x: x["market_data"].get(
                "price_change_percentage_24h", 0), reverse=True)

            output = "# 📊 Comparação de Performance (24h)\n\n"
            output += "| Posição | Moeda | Preço | Mudança 24h | Market Cap |\n"
            output += "|---------|-------|-------|-------------|------------|\n"

            for i, comp in enumerate(comparisons, 1):
                name = comp["name"]
                symbol = comp["symbol"].upper()
                price = comp["market_data"].get("current_price", 0)
                change = comp["market_data"].get(
                    "price_change_percentage_24h", 0)
                mcap = comp["market_data"].get("market_cap", 0)

                change_emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
                price_str = f"${price:.6f}" if price < 1 else f"${price:,.2f}"
                mcap_str = self.formatter._format_market_cap(
                    mcap, "usd") if mcap else "N/A"

                output += f"| {i}º | **{name}** ({symbol}) | {price_str} | {change_emoji} {change:+.2f}% | {mcap_str} |\n"

        elif metric == "technical":
            output = "# 🔍 Comparação Técnica\n\n"

            for comp in comparisons:
                name = comp["name"]
                symbol = comp["symbol"].upper()
                tech = comp["technical_data"]

                output += f"## {name} ({symbol})\n"

                if tech.get("rsi"):
                    rsi_status = tech.get("rsi_status", "neutral")
                    rsi_emoji = {"oversold": "🟢", "overbought": "🔴",
                                 "neutral": "🟡"}.get(rsi_status, "🟡")
                    output += f"• **RSI:** {tech['rsi']:.1f} {rsi_emoji} ({rsi_status})\n"

                if tech.get("macd_trend"):
                    macd_emoji = "🟢" if "bullish" in tech["macd_trend"] else "🔴"
                    output += f"• **MACD:** {tech['macd_trend'].replace('_', ' ')} {macd_emoji}\n"

                trend = tech.get("trend_direction", "uncertain")
                trend_emoji = {"bullish": "📈", "bearish": "📉",
                               "sideways": "↔️"}.get(trend, "❓")
                output += f"• **Tendência:** {trend_emoji} {trend.title()}\n\n"

        else:  # market_cap
            # Ordenar por market cap
            comparisons.sort(key=lambda x: x["market_data"].get(
                "market_cap", 0), reverse=True)

            output = "# 💰 Comparação por Market Cap\n\n"
            output += "| Ranking | Moeda | Market Cap | Volume 24h | Preço |\n"
            output += "|---------|-------|------------|------------|-------|\n"

            for i, comp in enumerate(comparisons, 1):
                name = comp["name"]
                symbol = comp["symbol"].upper()
                mcap = comp["market_data"].get("market_cap", 0)
                volume = comp["market_data"].get("volume_24h", 0)
                price = comp["market_data"].get("current_price", 0)

                mcap_str = self.formatter._format_market_cap(
                    mcap, "usd") if mcap else "N/A"
                volume_str = self.formatter._format_volume(
                    volume, "usd") if volume else "N/A"
                price_str = f"${price:.6f}" if price < 1 else f"${price:,.2f}"

                output += f"| {i}º | **{name}** ({symbol}) | {mcap_str} | {volume_str} | {price_str} |\n"

        output += "\n⚠️ *Esta é uma análise comparativa baseada em dados técnicos, não constitui aconselhamento financeiro*"
        return output
