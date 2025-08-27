#!/usr/bin/env python3
"""
Exemplo de Uso do Sistema de Output Padronizado
=============================================

Este script demonstra como usar o sistema de output padronizado
para análises de criptomoedas com o framework Agno.
"""

from agno_output_adapter import AgnoOutputAdapter
from output_formatter import OutputFormatter, AnalysisType, TrendDirection
from standard_crypto_toolkit import StandardCryptoAnalysisToolKit
import sys
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar path do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def exemplo_analise_completa():
    """Demonstra análise completa de uma criptomoeda"""

    print("🚀 === EXEMPLO: ANÁLISE COMPLETA DO BITCOIN ===\n")

    try:
        # Inicializar toolkit
        toolkit = StandardCryptoAnalysisToolKit()

        # Realizar análise completa
        resultado = toolkit.comprehensive_crypto_analysis(
            coin_id="bitcoin",
            include_sentiment=True,
            output_format="markdown"
        )

        print(resultado)

    except Exception as e:
        print(f"❌ Erro na análise completa: {e}")


def exemplo_resumo_rapido():
    """Demonstra resumo rápido de múltiplas moedas"""

    print("\n" + "="*60)
    print("📊 === EXEMPLO: RESUMOS RÁPIDOS ===\n")

    moedas = ["bitcoin", "ethereum", "cardano"]

    try:
        toolkit = StandardCryptoAnalysisToolKit()

        for moeda in moedas:
            print(f"--- {moeda.upper()} ---")
            resumo = toolkit.quick_crypto_summary(moeda)
            print(resumo)
            print()

    except Exception as e:
        print(f"❌ Erro nos resumos: {e}")


def exemplo_comparacao():
    """Demonstra comparação entre múltiplas criptomoedas"""

    print("\n" + "="*60)
    print("⚖️ === EXEMPLO: COMPARAÇÃO ENTRE CRIPTOMOEDAS ===\n")

    try:
        toolkit = StandardCryptoAnalysisToolKit()

        # Comparação por performance
        print("🏆 COMPARAÇÃO POR PERFORMANCE:\n")
        comparacao_perf = toolkit.multi_crypto_comparison(
            coin_ids=["bitcoin", "ethereum", "binancecoin", "cardano"],
            comparison_metric="performance"
        )
        print(comparacao_perf)

        print("\n" + "-"*40 + "\n")

        # Comparação técnica
        print("🔍 COMPARAÇÃO TÉCNICA:\n")
        comparacao_tech = toolkit.multi_crypto_comparison(
            coin_ids=["bitcoin", "ethereum", "solana"],
            comparison_metric="technical"
        )
        print(comparacao_tech)

    except Exception as e:
        print(f"❌ Erro na comparação: {e}")


def exemplo_formatos_diferentes():
    """Demonstra diferentes formatos de output"""

    print("\n" + "="*60)
    print("📋 === EXEMPLO: DIFERENTES FORMATOS DE OUTPUT ===\n")

    try:
        toolkit = StandardCryptoAnalysisToolKit()

        # Formato JSON
        print("📄 FORMATO JSON:\n")
        resultado_json = toolkit.comprehensive_crypto_analysis(
            coin_id="ethereum",
            include_sentiment=False,
            output_format="json"
        )
        print(resultado_json[:500] + "...\n")  # Apenas parte do JSON

        # Formato Summary
        print("📝 FORMATO RESUMO:\n")
        resultado_summary = toolkit.comprehensive_crypto_analysis(
            coin_id="ethereum",
            include_sentiment=False,
            output_format="summary"
        )
        print(resultado_summary)

    except Exception as e:
        print(f"❌ Erro nos formatos: {e}")


def exemplo_manual_formatter():
    """Demonstra uso manual do OutputFormatter"""

    print("\n" + "="*60)
    print("🛠️ === EXEMPLO: USO MANUAL DO FORMATTER ===\n")

    try:
        from output_formatter import (
            OutputFormatter, CryptoAnalysisResult, MarketData,
            TechnicalIndicators, TimeframeAnalysis, TimeframeType,
            TrendDirection, SentimentData
        )

        formatter = OutputFormatter()

        # Criar análise manual
        analysis = formatter.create_analysis_result(
            coin_id="bitcoin",
            coin_name="Bitcoin",
            coin_symbol="BTC",
            analysis_type=AnalysisType.TECHNICAL
        )

        # Preencher dados de exemplo
        analysis.market_data = MarketData(
            current_price=45000.00,
            price_change_24h=1200.50,
            price_change_percentage_24h=2.75,
            market_cap=850000000000,
            volume_24h=25000000000,
            market_cap_rank=1,
            currency="usd"
        )

        analysis.sentiment_data = SentimentData(
            overall_sentiment="positive",
            sentiment_score=0.65,
            news_count=12,
            key_topics=["adoption", "etf", "institutional"]
        )

        # Adicionar análise técnica
        tech_indicators = TechnicalIndicators(
            rsi=58.5,
            rsi_status="neutral",
            macd_line=850.25,
            macd_signal=720.40,
            macd_trend="bullish_crossover",
            sma_20=44500.00,
            sma_50=43200.00,
            sma_200=41800.00,
            moving_average_trend="golden_cross"
        )

        timeframe_analysis = TimeframeAnalysis(
            timeframe=TimeframeType.SHORT_TERM,
            period_days=30,
            trend_direction=TrendDirection.BULLISH,
            confidence_level=78.5,
            technical_indicators=tech_indicators,
            key_insights=[
                "Golden cross formado recentemente",
                "Volume acima da média em breakout",
                "Suporte forte em $44,000"
            ],
            trading_implications="Favorável para posições longas com stop em $43,500"
        )

        analysis.timeframe_analyses = [timeframe_analysis]
        analysis.overall_trend = TrendDirection.BULLISH
        analysis.confidence_score = 78.5
        analysis.key_takeaways = [
            "Tendência altista confirmada por múltiplos indicadores",
            "Golden cross sinaliza momento favorável",
            "Sentimento positivo suporta continuação da alta"
        ]
        analysis.risk_factors = [
            "Possível correção técnica em níveis de resistência",
            "Dependência de fatores macro-econômicos"
        ]
        analysis.opportunities = [
            "Entrada técnica favorável pós-breakout",
            "Momentum institucional crescente"
        ]

        # Gerar output formatado
        output_md = formatter.format_markdown_output(analysis)
        print(output_md)

    except Exception as e:
        print(f"❌ Erro no exemplo manual: {e}")


def main():
    """Executa todos os exemplos"""

    print("🔥 DEMONSTRAÇÃO DO SISTEMA DE OUTPUT PADRONIZADO")
    print("Framework Agno + Análises de Criptomoedas")
    print("=" * 60)

    # Verificar se as variáveis de ambiente estão configuradas
    if not os.getenv("COINGECKO_API_KEY"):
        print("⚠️ AVISO: COINGECKO_API_KEY não encontrada no ambiente")
        print("Alguns exemplos podem não funcionar corretamente.\n")

    try:
        # Exemplo 1: Análise completa
        exemplo_analise_completa()

        # Exemplo 2: Resumos rápidos
        exemplo_resumo_rapido()

        # Exemplo 3: Comparações
        exemplo_comparacao()

        # Exemplo 4: Formatos diferentes
        exemplo_formatos_diferentes()

        # Exemplo 5: Uso manual
        exemplo_manual_formatter()

        print("\n" + "="*60)
        print("✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
        print("Agora você pode integrar esse sistema ao seu agente Agno.")

    except KeyboardInterrupt:
        print("\n❌ Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro geral na execução: {e}")


if __name__ == "__main__":
    main()
