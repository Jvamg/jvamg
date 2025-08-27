"""
Output Formatter para Análises de Criptomoedas usando Framework Agno
===================================================================

Este módulo implementa um sistema padronizado de outputs para análises de criptomoedas,
garantindo consistência e estruturação dos resultados fornecidos pelo agente Agno.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class AnalysisType(Enum):
    """Tipos de análise disponíveis"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    COMPREHENSIVE = "comprehensive"
    PATTERN = "pattern"


class TimeframeType(Enum):
    """Tipos de timeframe para análise"""
    SHORT_TERM = "short_term"  # 1-30 days
    MEDIUM_TERM = "medium_term"  # 30-90 days
    LONG_TERM = "long_term"  # 90+ days


class TrendDirection(Enum):
    """Direções de tendência"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNCERTAIN = "uncertain"


@dataclass
class TechnicalIndicators:
    """Estrutura para indicadores técnicos"""
    rsi: Optional[float] = None
    rsi_status: Optional[str] = None  # "oversold", "overbought", "neutral"
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    # "bullish_crossover", "bearish_crossover", "neutral"
    macd_trend: Optional[str] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    # "golden_cross", "death_cross", "neutral"
    moving_average_trend: Optional[str] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


@dataclass
class MarketData:
    """Estrutura para dados de mercado"""
    current_price: float
    price_change_24h: float
    price_change_percentage_24h: float
    market_cap: Optional[float] = None
    market_cap_rank: Optional[int] = None
    volume_24h: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    ath: Optional[float] = None  # All-time high
    atl: Optional[float] = None  # All-time low
    currency: str = "usd"


@dataclass
class SentimentData:
    """Estrutura para análise de sentimento"""
    overall_sentiment: str  # "positive", "negative", "neutral", "mixed"
    sentiment_score: float  # -1 (very bearish) to +1 (very bullish)
    news_count: int
    key_topics: List[str]
    regulatory_impact: Optional[str] = None
    social_impact: Optional[str] = None


@dataclass
class TimeframeAnalysis:
    """Análise específica para um timeframe"""
    timeframe: TimeframeType
    period_days: int
    trend_direction: TrendDirection
    confidence_level: float  # 0-100
    technical_indicators: TechnicalIndicators
    key_insights: List[str]
    trading_implications: str


@dataclass
class CryptoAnalysisResult:
    """Resultado completo da análise de criptomoeda"""
    # Metadados
    timestamp: str
    coin_id: str
    coin_name: str
    coin_symbol: str
    analysis_type: AnalysisType

    # Dados principais
    market_data: MarketData
    sentiment_data: Optional[SentimentData] = None

    # Análises por timeframe
    timeframe_analyses: List[TimeframeAnalysis] = None

    # Resumo executivo
    overall_trend: TrendDirection = TrendDirection.UNCERTAIN
    confidence_score: float = 0.0  # 0-100
    key_takeaways: List[str] = None
    risk_factors: List[str] = None
    opportunities: List[str] = None

    # Recomendações
    short_term_outlook: str = ""
    long_term_outlook: str = ""

    # Disclaimer
    disclaimer: str = "Esta análise é baseada em dados históricos e técnicos. Não constitui aconselhamento financeiro."

    def __post_init__(self):
        if self.timeframe_analyses is None:
            self.timeframe_analyses = []
        if self.key_takeaways is None:
            self.key_takeaways = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.opportunities is None:
            self.opportunities = []


class OutputFormatter:
    """
    Formatador principal para outputs padronizados de análises de cripto
    """

    def __init__(self):
        self.version = "1.0.0"

    def create_analysis_result(
        self,
        coin_id: str,
        coin_name: str,
        coin_symbol: str,
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    ) -> CryptoAnalysisResult:
        """Cria uma estrutura base para resultado de análise"""

        return CryptoAnalysisResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            coin_id=coin_id,
            coin_name=coin_name,
            coin_symbol=coin_symbol,
            analysis_type=analysis_type,
            market_data=MarketData(
                current_price=0.0, price_change_24h=0.0, price_change_percentage_24h=0.0)
        )

    def format_markdown_output(self, analysis: CryptoAnalysisResult) -> str:
        """Gera output formatado em Markdown"""

        md = f"""# 📊 Análise de Criptomoeda: {analysis.coin_name} ({analysis.coin_symbol.upper()})

## 🔍 Resumo Executivo
**Tendência Geral:** {self._format_trend_with_emoji(analysis.overall_trend)}  
**Confiança:** {analysis.confidence_score:.1f}%  
**Timestamp:** {self._format_timestamp(analysis.timestamp)}

---

## 💰 Dados de Mercado

| Métrica | Valor |
|---------|-------|
| **Preço Atual** | {self._format_price(analysis.market_data.current_price, analysis.market_data.currency)} |
| **Mudança 24h** | {self._format_price_change(analysis.market_data.price_change_24h, analysis.market_data.price_change_percentage_24h, analysis.market_data.currency)} |"""

        if analysis.market_data.market_cap:
            md += f"\n| **Market Cap** | {self._format_market_cap(analysis.market_data.market_cap, analysis.market_data.currency)} |"

        if analysis.market_data.market_cap_rank:
            md += f"\n| **Ranking** | #{analysis.market_data.market_cap_rank} |"

        if analysis.market_data.volume_24h:
            md += f"\n| **Volume 24h** | {self._format_volume(analysis.market_data.volume_24h, analysis.market_data.currency)} |"

        # Análises por timeframe
        if analysis.timeframe_analyses:
            md += "\n\n---\n\n## 📈 Análise Técnica Multi-Timeframe\n"

            for tf_analysis in analysis.timeframe_analyses:
                md += f"\n### {self._format_timeframe_title(tf_analysis.timeframe)} ({tf_analysis.period_days} dias)\n"
                md += f"**Tendência:** {self._format_trend_with_emoji(tf_analysis.trend_direction)}  \n"
                md += f"**Confiança:** {tf_analysis.confidence_level:.1f}%\n\n"

                # Indicadores técnicos
                if tf_analysis.technical_indicators:
                    md += self._format_technical_indicators(
                        tf_analysis.technical_indicators)

                # Insights principais
                if tf_analysis.key_insights:
                    md += "**💡 Insights Principais:**\n"
                    for insight in tf_analysis.key_insights:
                        md += f"- {insight}\n"
                    md += "\n"

                # Implicações para trading
                if tf_analysis.trading_implications:
                    md += f"**🎯 Implicações para Trading:** {tf_analysis.trading_implications}\n\n"

        # Análise de sentimento
        if analysis.sentiment_data:
            md += "---\n\n## 📰 Análise de Sentimento\n"
            md += f"**Sentimento Geral:** {self._format_sentiment_with_emoji(analysis.sentiment_data.overall_sentiment)}  \n"
            md += f"**Score de Sentimento:** {analysis.sentiment_data.sentiment_score:.2f} (-1 a +1)  \n"
            md += f"**Notícias Analisadas:** {analysis.sentiment_data.news_count}\n\n"

            if analysis.sentiment_data.key_topics:
                md += "**🔥 Tópicos em Destaque:**\n"
                for topic in analysis.sentiment_data.key_topics:
                    md += f"- {topic}\n"
                md += "\n"

        # Resumo de takeaways
        if analysis.key_takeaways:
            md += "---\n\n## 🎯 Principais Conclusões\n"
            for takeaway in analysis.key_takeaways:
                md += f"✅ {takeaway}\n"
            md += "\n"

        # Riscos e oportunidades
        if analysis.risk_factors or analysis.opportunities:
            md += "---\n\n## ⚖️ Riscos e Oportunidades\n"

            if analysis.risk_factors:
                md += "### ⚠️ Fatores de Risco\n"
                for risk in analysis.risk_factors:
                    md += f"- {risk}\n"
                md += "\n"

            if analysis.opportunities:
                md += "### 🚀 Oportunidades\n"
                for opportunity in analysis.opportunities:
                    md += f"- {opportunity}\n"
                md += "\n"

        # Outlook
        if analysis.short_term_outlook or analysis.long_term_outlook:
            md += "---\n\n## 🔮 Perspectivas\n"

            if analysis.short_term_outlook:
                md += f"**📊 Curto Prazo:** {analysis.short_term_outlook}\n\n"

            if analysis.long_term_outlook:
                md += f"**📈 Longo Prazo:** {analysis.long_term_outlook}\n\n"

        # Disclaimer
        md += f"---\n\n## ⚠️ Aviso Importante\n{analysis.disclaimer}\n"

        return md

    def format_json_output(self, analysis: CryptoAnalysisResult) -> str:
        """Gera output formatado em JSON"""
        return json.dumps(asdict(analysis), indent=2, ensure_ascii=False)

    def format_compact_summary(self, analysis: CryptoAnalysisResult) -> str:
        """Gera um resumo compacto da análise"""

        price_emoji = "🟢" if analysis.market_data.price_change_percentage_24h > 0 else "🔴" if analysis.market_data.price_change_percentage_24h < 0 else "🟡"
        trend_emoji = self._get_trend_emoji(analysis.overall_trend)

        summary = f"{trend_emoji} **{analysis.coin_symbol.upper()}** | "
        summary += f"{self._format_price(analysis.market_data.current_price, analysis.market_data.currency)} "
        summary += f"({price_emoji} {analysis.market_data.price_change_percentage_24h:+.2f}%) | "
        summary += f"Tendência: {analysis.overall_trend.value.title()} | "
        summary += f"Confiança: {analysis.confidence_score:.0f}%"

        return summary

    # Métodos auxiliares para formatação

    def _format_trend_with_emoji(self, trend: TrendDirection) -> str:
        """Formata tendência com emoji"""
        emoji_map = {
            TrendDirection.BULLISH: "📈 Alta",
            TrendDirection.BEARISH: "📉 Baixa",
            TrendDirection.SIDEWAYS: "↔️ Lateral",
            TrendDirection.UNCERTAIN: "❓ Incerta"
        }
        return emoji_map.get(trend, "❓ Incerta")

    def _get_trend_emoji(self, trend: TrendDirection) -> str:
        """Retorna emoji da tendência"""
        emoji_map = {
            TrendDirection.BULLISH: "📈",
            TrendDirection.BEARISH: "📉",
            TrendDirection.SIDEWAYS: "↔️",
            TrendDirection.UNCERTAIN: "❓"
        }
        return emoji_map.get(trend, "❓")

    def _format_sentiment_with_emoji(self, sentiment: str) -> str:
        """Formata sentimento com emoji"""
        emoji_map = {
            "positive": "😊 Positivo",
            "negative": "😰 Negativo",
            "neutral": "😐 Neutro",
            "mixed": "🤔 Misto"
        }
        return emoji_map.get(sentiment.lower(), "😐 Neutro")

    def _format_timeframe_title(self, timeframe: TimeframeType) -> str:
        """Formata título do timeframe"""
        title_map = {
            TimeframeType.SHORT_TERM: "📊 Curto Prazo",
            TimeframeType.MEDIUM_TERM: "📈 Médio Prazo",
            TimeframeType.LONG_TERM: "🎯 Longo Prazo"
        }
        return title_map.get(timeframe, "📊 Análise")

    def _format_price(self, price: float, currency: str) -> str:
        """Formata preço com símbolo da moeda"""
        symbol_map = {
            "usd": "$",
            "eur": "€",
            "brl": "R$",
            "btc": "₿"
        }
        symbol = symbol_map.get(currency.lower(), "$")

        if price >= 1:
            return f"{symbol}{price:,.2f}"
        else:
            return f"{symbol}{price:.6f}"

    def _format_price_change(self, change: float, percentage: float, currency: str) -> str:
        """Formata mudança de preço"""
        symbol_map = {
            "usd": "$",
            "eur": "€",
            "brl": "R$",
            "btc": "₿"
        }
        symbol = symbol_map.get(currency.lower(), "$")

        sign = "+" if change > 0 else ""
        emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"

        return f"{emoji} {sign}{symbol}{change:.2f} ({sign}{percentage:.2f}%)"

    def _format_market_cap(self, market_cap: float, currency: str) -> str:
        """Formata market cap"""
        symbol_map = {
            "usd": "$",
            "eur": "€",
            "brl": "R$"
        }
        symbol = symbol_map.get(currency.lower(), "$")

        if market_cap >= 1e12:
            return f"{symbol}{market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"{symbol}{market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"{symbol}{market_cap/1e6:.2f}M"
        else:
            return f"{symbol}{market_cap:,.0f}"

    def _format_volume(self, volume: float, currency: str) -> str:
        """Formata volume"""
        return self._format_market_cap(volume, currency)

    def _format_timestamp(self, timestamp_str: str) -> str:
        """Formata timestamp"""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime("%d/%m/%Y às %H:%M UTC")
        except:
            return timestamp_str

    def _format_technical_indicators(self, indicators: TechnicalIndicators) -> str:
        """Formata indicadores técnicos"""
        md = "**📊 Indicadores Técnicos:**\n"

        if indicators.rsi is not None:
            rsi_status_emoji = {
                "oversold": "🟢",
                "overbought": "🔴",
                "neutral": "🟡"
            }
            emoji = rsi_status_emoji.get(indicators.rsi_status, "🟡")
            md += f"- **RSI:** {indicators.rsi:.2f} {emoji} ({indicators.rsi_status or 'neutral'})\n"

        if indicators.macd_line is not None:
            macd_emoji = "🟢" if (indicators.macd_line or 0) > (
                indicators.macd_signal or 0) else "🔴"
            md += f"- **MACD:** {indicators.macd_line:.4f} / Signal: {indicators.macd_signal:.4f} {macd_emoji}\n"

        if indicators.sma_20 is not None:
            md += f"- **SMA 20:** {indicators.sma_20:.2f}\n"
        if indicators.sma_50 is not None:
            md += f"- **SMA 50:** {indicators.sma_50:.2f}\n"
        if indicators.sma_200 is not None:
            md += f"- **SMA 200:** {indicators.sma_200:.2f}\n"

        if indicators.moving_average_trend:
            trend_emoji = "🟢" if "golden" in indicators.moving_average_trend.lower(
            ) else "🔴" if "death" in indicators.moving_average_trend.lower() else "🟡"
            md += f"- **Tendência MA:** {indicators.moving_average_trend} {trend_emoji}\n"

        md += "\n"
        return md


# Funções auxiliares para integração com o agente

def parse_technical_analysis_response(response_text: str) -> Dict[str, Any]:
    """
    Extrai dados de análise técnica de uma resposta do agente
    Esta função pode ser customizada conforme o formato de resposta do seu agente
    """
    # Implementação placeholder - adapte conforme necessário
    return {
        "rsi": None,
        "macd_line": None,
        "macd_signal": None,
        "sma_20": None,
        "sma_50": None,
        "sma_200": None
    }


def create_standard_crypto_analysis(
    coin_id: str,
    coin_name: str,
    coin_symbol: str,
    technical_data: Dict = None,
    market_data_dict: Dict = None,
    sentiment_data_dict: Dict = None
) -> CryptoAnalysisResult:
    """
    Função helper para criar análise padronizada a partir de dados do agente
    """
    formatter = OutputFormatter()
    analysis = formatter.create_analysis_result(
        coin_id, coin_name, coin_symbol)

    # Preencher dados de mercado
    if market_data_dict:
        analysis.market_data.current_price = market_data_dict.get(
            "current_price", 0.0)
        analysis.market_data.price_change_24h = market_data_dict.get(
            "price_change_24h", 0.0)
        analysis.market_data.price_change_percentage_24h = market_data_dict.get(
            "price_change_percentage_24h", 0.0)
        analysis.market_data.market_cap = market_data_dict.get("market_cap")
        analysis.market_data.volume_24h = market_data_dict.get("volume_24h")
        analysis.market_data.market_cap_rank = market_data_dict.get(
            "market_cap_rank")
        analysis.market_data.currency = market_data_dict.get("currency", "usd")

    # Preencher dados de sentimento
    if sentiment_data_dict:
        analysis.sentiment_data = SentimentData(
            overall_sentiment=sentiment_data_dict.get(
                "overall_sentiment", "neutral"),
            sentiment_score=sentiment_data_dict.get("sentiment_score", 0.0),
            news_count=sentiment_data_dict.get("news_count", 0),
            key_topics=sentiment_data_dict.get("key_topics", [])
        )

    return analysis
