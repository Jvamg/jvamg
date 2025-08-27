"""
Agno Output Adapter
==================

Adaptador para integrar o sistema de Output Formatter padronizado 
com o agente Agno existente para análises de criptomoedas.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from output_formatter import (
    OutputFormatter, CryptoAnalysisResult, TechnicalIndicators,
    MarketData, SentimentData, TimeframeAnalysis, AnalysisType,
    TimeframeType, TrendDirection, create_standard_crypto_analysis
)


class AgnoOutputAdapter:
    """
    Adaptador que converte outputs do agente Agno para formato padronizado
    """

    def __init__(self):
        self.formatter = OutputFormatter()

    def parse_agno_technical_analysis(self, technical_response: str) -> Dict[str, Any]:
        """
        Extrai indicadores técnicos de uma resposta do perform_technical_analysis()
        """
        indicators = {
            "rsi": None,
            "rsi_status": None,
            "macd_line": None,
            "macd_signal": None,
            "macd_histogram": None,
            "macd_trend": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "moving_average_trend": None,
            "trend_direction": "uncertain",
            "confidence": 50.0
        }

        try:
            # Extrair RSI - regex para formato "**RSI (14)**: 41.15"
            rsi_match = re.search(r'RSI\s*\([0-9]+\)\*\*:\s*([0-9]+\.?[0-9]*)', 
                                  technical_response, re.IGNORECASE)
            if rsi_match:
                rsi_value = float(rsi_match.group(1))
                indicators["rsi"] = rsi_value

                # Determinar status do RSI
                if rsi_value > 70:
                    indicators["rsi_status"] = "overbought"
                elif rsi_value < 30:
                    indicators["rsi_status"] = "oversold"
                else:
                    indicators["rsi_status"] = "neutral"

            # Extrair MACD
            macd_match = re.search(
                r'MACD.*?(-?\d+\.?\d*)', technical_response, re.IGNORECASE)
            if macd_match:
                indicators["macd_line"] = float(macd_match.group(1))

            signal_match = re.search(
                r'Signal.*?(-?\d+\.?\d*)', technical_response, re.IGNORECASE)
            if signal_match:
                indicators["macd_signal"] = float(signal_match.group(1))

            # Determinar tendência MACD
            if indicators["macd_line"] and indicators["macd_signal"]:
                if indicators["macd_line"] > indicators["macd_signal"]:
                    indicators["macd_trend"] = "bullish_crossover"
                else:
                    indicators["macd_trend"] = "bearish_crossover"

            # Extrair SMAs - corrigido para lidar com formatação monetária ($ e vírgulas)
            sma_patterns = [
                (r'SMA\s*20.*?\$?([0-9,]+\.?\d*)', "sma_20"),
                (r'SMA\s*50.*?\$?([0-9,]+\.?\d*)', "sma_50"),
                (r'SMA\s*200.*?\$?([0-9,]+\.?\d*)', "sma_200")
            ]

            for pattern, key in sma_patterns:
                match = re.search(pattern, technical_response, re.IGNORECASE)
                if match:
                    # Remover vírgulas da formatação antes de converter para float
                    value_str = match.group(1).replace(',', '')
                    indicators[key] = float(value_str)

            # Detectar Golden Cross / Death Cross
            if "golden cross" in technical_response.lower():
                indicators["moving_average_trend"] = "golden_cross"
            elif "death cross" in technical_response.lower():
                indicators["moving_average_trend"] = "death_cross"

            # Extrair direção geral da tendência
            if any(word in technical_response.lower() for word in ["bullish", "alta", "positiv"]):
                indicators["trend_direction"] = "bullish"
            elif any(word in technical_response.lower() for word in ["bearish", "baixa", "negativ"]):
                indicators["trend_direction"] = "bearish"
            elif any(word in technical_response.lower() for word in ["sideways", "lateral", "neutro"]):
                indicators["trend_direction"] = "sideways"

            # Tentar extrair score de confiança
            confidence_match = re.search(
                r'confiança.*?(\d+)', technical_response, re.IGNORECASE)
            if confidence_match:
                indicators["confidence"] = float(confidence_match.group(1))
            elif "strong" in technical_response.lower() or "forte" in technical_response.lower():
                indicators["confidence"] = 80.0
            elif "weak" in technical_response.lower() or "fraco" in technical_response.lower():
                indicators["confidence"] = 30.0

        except Exception as e:
            print(f"Error parsing technical analysis: {e}")

        return indicators

    def parse_market_data_response(self, market_response: str) -> Dict[str, Any]:
        """
        Extrai dados de mercado de uma resposta do get_market_data()
        """
        market_data = {}

        try:
            # Extrair preço atual
            price_patterns = [
                r'\$(\d+,?\d*\.?\d*)',
                r'price.*?(\d+\.?\d*)',
                r'preço.*?(\d+\.?\d*)'
            ]

            for pattern in price_patterns:
                match = re.search(pattern, market_response, re.IGNORECASE)
                if match:
                    price_str = match.group(1).replace(',', '')
                    market_data["current_price"] = float(price_str)
                    break

            # Extrair mudança percentual
            percentage_match = re.search(r'([-+]?\d+\.?\d*)%', market_response)
            if percentage_match:
                market_data["price_change_percentage_24h"] = float(
                    percentage_match.group(1))

            # Extrair market cap - corrigido para capturar com vírgulas e incluir T (trilhões)
            mcap_match = re.search(
                r'market cap.*?\$([0-9,]+\.?\d*[BTMK]?)', market_response, re.IGNORECASE)
            if mcap_match:
                mcap_str = mcap_match.group(1)
                multiplier = 1
                if mcap_str.upper().endswith('T'):
                    multiplier = 1e12
                elif mcap_str.upper().endswith('B'):
                    multiplier = 1e9
                elif mcap_str.upper().endswith('M'):
                    multiplier = 1e6
                elif mcap_str.upper().endswith('K'):
                    multiplier = 1e3

                mcap_value = float(mcap_str.rstrip('BTMKbtmk').replace(',', ''))
                market_data["market_cap"] = mcap_value * multiplier

            # Extrair volume - corrigido para capturar com vírgulas e incluir T (trilhões)  
            volume_match = re.search(
                r'volume.*?\$([0-9,]+\.?\d*[BTMK]?)', market_response, re.IGNORECASE)
            if volume_match:
                volume_str = volume_match.group(1)
                multiplier = 1
                if volume_str.upper().endswith('T'):
                    multiplier = 1e12
                elif volume_str.upper().endswith('B'):
                    multiplier = 1e9
                elif volume_str.upper().endswith('M'):
                    multiplier = 1e6
                elif volume_str.upper().endswith('K'):
                    multiplier = 1e3

                volume_value = float(volume_str.rstrip('BTMKbtmk').replace(',', ''))
                market_data["volume_24h"] = volume_value * multiplier

            # Extrair ranking
            rank_match = re.search(
                r'rank.*?#?(\d+)', market_response, re.IGNORECASE)
            if rank_match:
                market_data["market_cap_rank"] = int(rank_match.group(1))

        except Exception as e:
            print(f"Error parsing market data: {e}")

        return market_data

    def parse_sentiment_analysis(self, news_response: str, articles_count: int = 0) -> Dict[str, Any]:
        """
        Extrai análise de sentimento de uma resposta com notícias
        """
        sentiment_data = {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "news_count": articles_count,
            "key_topics": []
        }

        try:
            # Detectar sentimento geral
            positive_words = ["positive", "bullish",
                              "optimistic", "positivo", "otimista", "alta"]
            negative_words = ["negative", "bearish",
                              "pessimistic", "negativo", "pessimista", "baixa"]

            response_lower = news_response.lower()

            positive_count = sum(
                1 for word in positive_words if word in response_lower)
            negative_count = sum(
                1 for word in negative_words if word in response_lower)

            if positive_count > negative_count:
                sentiment_data["overall_sentiment"] = "positive"
                sentiment_data["sentiment_score"] = min(
                    0.8, (positive_count - negative_count) * 0.2)
            elif negative_count > positive_count:
                sentiment_data["overall_sentiment"] = "negative"
                sentiment_data["sentiment_score"] = max(
                    -0.8, (positive_count - negative_count) * 0.2)
            elif positive_count == negative_count and positive_count > 0:
                sentiment_data["overall_sentiment"] = "mixed"
                sentiment_data["sentiment_score"] = 0.0

            # Extrair tópicos chave
            topics_patterns = [
                r'regulat\w+',
                r'bitcoin|btc',
                r'ethereum|eth',
                r'adoption',
                r'partnership',
                r'upgrade',
                r'fork',
                r'sec',
                r'etf'
            ]

            for pattern in topics_patterns:
                matches = re.findall(pattern, response_lower)
                if matches:
                    sentiment_data["key_topics"].extend(
                        matches[:2])  # Limitar a 2 por padrão

            # Remover duplicatas
            sentiment_data["key_topics"] = list(set(sentiment_data["key_topics"]))[
                :5]  # Max 5 tópicos

        except Exception as e:
            print(f"Error parsing sentiment analysis: {e}")

        return sentiment_data

    def create_timeframe_analysis(
        self,
        timeframe: TimeframeType,
        period_days: int,
        technical_data: Dict[str, Any],
        insights: List[str] = None
    ) -> TimeframeAnalysis:
        """
        Cria análise estruturada para um timeframe específico
        """
        # Mapear trend_direction string para enum
        trend_map = {
            "bullish": TrendDirection.BULLISH,
            "bearish": TrendDirection.BEARISH,
            "sideways": TrendDirection.SIDEWAYS,
            "uncertain": TrendDirection.UNCERTAIN
        }

        trend_direction = trend_map.get(technical_data.get(
            "trend_direction", "uncertain"), TrendDirection.UNCERTAIN)

        # Criar indicadores técnicos
        technical_indicators = TechnicalIndicators(
            rsi=technical_data.get("rsi"),
            rsi_status=technical_data.get("rsi_status"),
            macd_line=technical_data.get("macd_line"),
            macd_signal=technical_data.get("macd_signal"),
            macd_histogram=technical_data.get("macd_histogram"),
            macd_trend=technical_data.get("macd_trend"),
            sma_20=technical_data.get("sma_20"),
            sma_50=technical_data.get("sma_50"),
            sma_200=technical_data.get("sma_200"),
            moving_average_trend=technical_data.get("moving_average_trend")
        )

        # Gerar implicações de trading baseadas no timeframe
        trading_implications = self._generate_trading_implications(
            timeframe, trend_direction, technical_indicators)

        return TimeframeAnalysis(
            timeframe=timeframe,
            period_days=period_days,
            trend_direction=trend_direction,
            confidence_level=technical_data.get("confidence", 50.0),
            technical_indicators=technical_indicators,
            key_insights=insights or [],
            trading_implications=trading_implications
        )

    def _generate_trading_implications(
        self,
        timeframe: TimeframeType,
        trend: TrendDirection,
        indicators: TechnicalIndicators
    ) -> str:
        """
        Gera implicações de trading baseadas no timeframe e indicadores
        """
        implications = {
            TimeframeType.SHORT_TERM: {
                TrendDirection.BULLISH: "Favorável para day trading e scalping. Considere posições longas com stop-loss apertado.",
                TrendDirection.BEARISH: "Risco elevado para posições longas. Considere aguardar ou operar vendido.",
                TrendDirection.SIDEWAYS: "Range trading. Comprar no suporte e vender na resistência.",
                TrendDirection.UNCERTAIN: "Mercado indeciso. Aguarde confirmação de direção antes de posicionar."
            },
            TimeframeType.MEDIUM_TERM: {
                TrendDirection.BULLISH: "Oportunidade para swing trading. Posições de 1-4 semanas podem ser lucrativas.",
                TrendDirection.BEARISH: "Evite posições longas médio prazo. Considere estratégias defensivas.",
                TrendDirection.SIDEWAYS: "Mercado consolidando. Aguarde breakout para posicionar swing trades.",
                TrendDirection.UNCERTAIN: "Aguarde definição de tendência. Mantenha posições pequenas."
            },
            TimeframeType.LONG_TERM: {
                TrendDirection.BULLISH: "Favorável para HODLing e DCA. Tendência de longo prazo suporta acumulação.",
                TrendDirection.BEARISH: "Bear market. Considere aguardar melhores pontos de entrada a longo prazo.",
                TrendDirection.SIDEWAYS: "Acumulação/distribuição. Foque em fundamentals para decisões de longo prazo.",
                TrendDirection.UNCERTAIN: "Mercado em transição. Diversifique e mantenha estratégia conservadora."
            }
        }

        base_implication = implications.get(
            timeframe, {}).get(trend, "Análise inconclusiva.")

        # Adicionar detalhes baseados nos indicadores
        if indicators.rsi_status == "oversold" and trend == TrendDirection.BEARISH:
            base_implication += " RSI indica possível reversão técnica de curto prazo."
        elif indicators.rsi_status == "overbought" and trend == TrendDirection.BULLISH:
            base_implication += " RSI sugere possível correção técnica no curto prazo."

        if indicators.moving_average_trend == "golden_cross":
            base_implication += " Golden cross fortalece perspectiva altista."
        elif indicators.moving_average_trend == "death_cross":
            base_implication += " Death cross reforça perspectiva baixista."

        return base_implication

    def create_comprehensive_analysis(
        self,
        coin_id: str,
        coin_name: str,
        coin_symbol: str,
        market_response: str,
        short_term_technical: str,
        long_term_technical: str,
        news_response: str = "",
        news_count: int = 0
    ) -> CryptoAnalysisResult:
        """
        Cria análise abrangente combinando todas as respostas do agente
        """
        # Parsear dados de mercado
        market_data_dict = self.parse_market_data_response(market_response)

        # Parsear análises técnicas
        short_tech_data = self.parse_agno_technical_analysis(
            short_term_technical)
        long_tech_data = self.parse_agno_technical_analysis(
            long_term_technical)

        # Parsear sentimento (se disponível)
        sentiment_data_dict = None
        if news_response:
            sentiment_data_dict = self.parse_sentiment_analysis(
                news_response, news_count)

        # Criar análise base
        analysis = create_standard_crypto_analysis(
            coin_id, coin_name, coin_symbol,
            market_data_dict=market_data_dict,
            sentiment_data_dict=sentiment_data_dict
        )

        # Adicionar análises por timeframe
        short_term_analysis = self.create_timeframe_analysis(
            TimeframeType.SHORT_TERM,
            30,
            short_tech_data,
            self._extract_insights(short_term_technical)
        )

        long_term_analysis = self.create_timeframe_analysis(
            TimeframeType.LONG_TERM,
            365,
            long_tech_data,
            self._extract_insights(long_term_technical)
        )

        analysis.timeframe_analyses = [short_term_analysis, long_term_analysis]

        # Determinar tendência geral e confiança
        analysis.overall_trend = self._determine_overall_trend(
            short_tech_data, long_tech_data)
        analysis.confidence_score = self._calculate_confidence_score(
            short_tech_data, long_tech_data, sentiment_data_dict)

        # Gerar takeaways, riscos e oportunidades
        analysis.key_takeaways = self._generate_key_takeaways(analysis)
        analysis.risk_factors = self._generate_risk_factors(analysis)
        analysis.opportunities = self._generate_opportunities(analysis)

        # Gerar outlooks
        analysis.short_term_outlook = self._generate_short_term_outlook(
            short_term_analysis, sentiment_data_dict)
        analysis.long_term_outlook = self._generate_long_term_outlook(
            long_term_analysis)

        return analysis

    def _extract_insights(self, technical_response: str) -> List[str]:
        """
        Extrai insights principais de uma resposta técnica
        """
        insights = []

        # Padrões para identificar insights
        insight_patterns = [
            r'strong (bullish|bearish)',
            r'golden cross',
            r'death cross',
            r'oversold',
            r'overbought',
            r'breakout',
            r'breakdown',
            r'resistance',
            r'support'
        ]

        for pattern in insight_patterns:
            matches = re.findall(pattern, technical_response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    insights.append(f"Padrão identificado: {' '.join(match)}")
                else:
                    insights.append(f"Padrão identificado: {match}")

        return insights[:3]  # Limitar a 3 insights principais

    def _determine_overall_trend(self, short_data: Dict, long_data: Dict) -> TrendDirection:
        """
        Determina tendência geral combinando análises de curto e longo prazo
        """
        short_trend = short_data.get("trend_direction", "uncertain")
        long_trend = long_data.get("trend_direction", "uncertain")

        trend_map = {
            "bullish": TrendDirection.BULLISH,
            "bearish": TrendDirection.BEARISH,
            "sideways": TrendDirection.SIDEWAYS,
            "uncertain": TrendDirection.UNCERTAIN
        }

        short_enum = trend_map.get(short_trend, TrendDirection.UNCERTAIN)
        long_enum = trend_map.get(long_trend, TrendDirection.UNCERTAIN)

        # Lógica de combinação: longo prazo tem mais peso
        if long_enum == short_enum:
            return long_enum
        elif long_enum != TrendDirection.UNCERTAIN:
            return long_enum  # Priorizar longo prazo se não for incerto
        else:
            return short_enum  # Usar curto prazo se longo for incerto

    def _calculate_confidence_score(self, short_data: Dict, long_data: Dict, sentiment_data: Dict = None) -> float:
        """
        Calcula score de confiança baseado na convergência dos indicadores
        """
        base_confidence = 50.0

        # Boost se tendências convergem
        if short_data.get("trend_direction") == long_data.get("trend_direction"):
            base_confidence += 20.0

        # Boost se múltiplos indicadores concordam
        short_indicators = [short_data.get(k) for k in [
            "rsi_status", "macd_trend", "moving_average_trend"] if short_data.get(k)]
        if len(short_indicators) >= 2:
            base_confidence += 10.0

        # Boost se sentimento alinha com técnica
        if sentiment_data:
            sentiment = sentiment_data.get("overall_sentiment", "neutral")
            short_trend = short_data.get("trend_direction", "uncertain")

            if (sentiment == "positive" and short_trend == "bullish") or \
               (sentiment == "negative" and short_trend == "bearish"):
                base_confidence += 15.0
            elif sentiment == "mixed":
                base_confidence -= 5.0  # Penalizar sentimento misto

        return min(95.0, max(20.0, base_confidence))  # Limitar entre 20-95%

    def _generate_key_takeaways(self, analysis: CryptoAnalysisResult) -> List[str]:
        """
        Gera principais conclusões da análise
        """
        takeaways = []

        # Takeaway sobre preço
        price_change = analysis.market_data.price_change_percentage_24h
        if price_change > 5:
            takeaways.append(
                f"Forte alta de {price_change:.1f}% nas últimas 24h indica momentum positivo")
        elif price_change < -5:
            takeaways.append(
                f"Correção de {abs(price_change):.1f}% nas últimas 24h pode indicar oportunidade de compra")

        # Takeaway sobre tendência
        if analysis.overall_trend != TrendDirection.UNCERTAIN:
            takeaways.append(
                f"Tendência geral {analysis.overall_trend.value} com {analysis.confidence_score:.0f}% de confiança")

        # Takeaway sobre indicadores
        if analysis.timeframe_analyses:
            for tf_analysis in analysis.timeframe_analyses:
                if tf_analysis.technical_indicators.rsi_status in ["oversold", "overbought"]:
                    takeaways.append(
                        f"RSI {tf_analysis.timeframe.value.replace('_', ' ')} indica mercado {tf_analysis.technical_indicators.rsi_status}")

                if tf_analysis.technical_indicators.moving_average_trend:
                    takeaways.append(
                        f"Médias móveis sugerem {tf_analysis.technical_indicators.moving_average_trend.replace('_', ' ')}")

        return takeaways[:4]  # Máximo 4 takeaways

    def _generate_risk_factors(self, analysis: CryptoAnalysisResult) -> List[str]:
        """
        Gera fatores de risco identificados
        """
        risks = []

        # Riscos técnicos
        if analysis.timeframe_analyses:
            for tf_analysis in analysis.timeframe_analyses:
                if tf_analysis.technical_indicators.rsi_status == "overbought":
                    risks.append(
                        "RSI em sobrecompra sugere possível correção técnica")

                if tf_analysis.trend_direction == TrendDirection.BEARISH:
                    risks.append(
                        f"Tendência {tf_analysis.timeframe.value.replace('_', ' ')} baixista aumenta risco de perdas")

        # Riscos de sentimento
        if analysis.sentiment_data:
            if analysis.sentiment_data.overall_sentiment == "negative":
                risks.append(
                    "Sentimento de mercado negativo pode pressionar preços")
            elif analysis.sentiment_data.overall_sentiment == "mixed":
                risks.append(
                    "Sentimento misto indica incerteza e possível volatilidade")

        # Risco de confiança baixa
        if analysis.confidence_score < 60:
            risks.append(
                "Baixa convergência de sinais aumenta incerteza da análise")

        return risks[:3]  # Máximo 3 riscos

    def _generate_opportunities(self, analysis: CryptoAnalysisResult) -> List[str]:
        """
        Gera oportunidades identificadas
        """
        opportunities = []

        # Oportunidades técnicas
        if analysis.timeframe_analyses:
            for tf_analysis in analysis.timeframe_analyses:
                if tf_analysis.technical_indicators.rsi_status == "oversold":
                    opportunities.append(
                        "RSI em sobrevenda indica possível oportunidade de compra técnica")

                if tf_analysis.technical_indicators.moving_average_trend == "golden_cross":
                    opportunities.append(
                        "Golden cross sugere início de tendência altista sustentável")

                if tf_analysis.trend_direction == TrendDirection.BULLISH:
                    opportunities.append(
                        f"Tendência {tf_analysis.timeframe.value.replace('_', ' ')} altista favorece posições longas")

        # Oportunidades de sentimento
        if analysis.sentiment_data:
            if analysis.sentiment_data.overall_sentiment == "positive":
                opportunities.append(
                    "Sentimento positivo pode impulsionar valorização")

        return opportunities[:3]  # Máximo 3 oportunidades

    def _generate_short_term_outlook(self, short_analysis: TimeframeAnalysis, sentiment_data: Dict = None) -> str:
        """
        Gera perspectiva de curto prazo
        """
        base_outlook = {
            TrendDirection.BULLISH: "Perspectiva positiva para os próximos dias/semanas",
            TrendDirection.BEARISH: "Pressão vendedora pode continuar no curto prazo",
            TrendDirection.SIDEWAYS: "Consolidação esperada com movimentos laterais",
            TrendDirection.UNCERTAIN: "Direção indefinida requer cautela"
        }

        outlook = base_outlook.get(
            short_analysis.trend_direction, "Análise inconclusiva")

        # Adicionar contexto do RSI se disponível
        if short_analysis.technical_indicators.rsi_status:
            rsi_context = {
                "oversold": " com possível recuperação técnica",
                "overbought": " mas atenção a possíveis correções",
                "neutral": " em zona neutra de momentum"
            }
            outlook += rsi_context.get(
                short_analysis.technical_indicators.rsi_status, "")

        return outlook

    def _generate_long_term_outlook(self, long_analysis: TimeframeAnalysis) -> str:
        """
        Gera perspectiva de longo prazo
        """
        base_outlook = {
            TrendDirection.BULLISH: "Fundamentals técnicos suportam valorização sustentável",
            TrendDirection.BEARISH: "Tendência de longo prazo sugere cautela para investimentos",
            TrendDirection.SIDEWAYS: "Período de consolidação pode definir próxima grande movimentação",
            TrendDirection.UNCERTAIN: "Indefinição requer acompanhamento de fatores fundamentais"
        }

        outlook = base_outlook.get(
            long_analysis.trend_direction, "Análise de longo prazo inconclusiva")

        # Adicionar contexto das médias móveis
        if long_analysis.technical_indicators.moving_average_trend:
            ma_context = {
                "golden_cross": " com médias móveis em configuração muito positiva",
                "death_cross": " com médias móveis sinalizando bear market",
                "neutral": " com médias móveis em configuração neutra"
            }
            outlook += ma_context.get(
                long_analysis.technical_indicators.moving_average_trend, "")

        return outlook
