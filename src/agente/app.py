from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv
import os
import uuid
import json
import argparse
import time
import re
import sys
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
from agno.tools.googlesearch import GoogleSearchTools
from coingeckoToolKit import CoinGeckoToolKit
from coindeskToolKit import CoinDeskToolKit
from fearGreedToolKit import FearGreedToolKit

load_dotenv()


# Pydantic model for crypto analysis with Fear & Greed integration
class CryptoAnalysis(BaseModel):
    """Complete cryptocurrency analysis structure"""
    
    type: str = Field(default="comprehensive_crypto_analysis", description="Analysis type")
    summary: str = Field(..., description="Executive summary of findings")
    
    # Price data
    price_current: float = Field(..., description="Current price")
    price_change_24h: float = Field(..., description="24h price change %") 
    price_trend: str = Field(..., description="Price trend: bullish/bearish/neutral")
    volume_24h: Optional[float] = Field(None, description="24h trading volume")
    
    # Technical analysis
    rsi: Optional[float] = Field(None, description="RSI value (0-100)")
    macd_signal: Optional[str] = Field(None, description="MACD signal")
    sma_20: Optional[float] = Field(None, description="SMA 20")
    sma_50: Optional[float] = Field(None, description="SMA 50")
    technical_signal: str = Field(..., description="Technical signal: buy/sell/hold")
    
    # Support and resistance
    resistance_levels: List[float] = Field(default_factory=list, description="Resistance levels")
    support_levels: List[float] = Field(default_factory=list, description="Support levels")
    levels_confidence: Optional[float] = Field(None, description="Confidence 0.0-1.0")
    
    # Market sentiment with Fear & Greed
    news_sentiment: str = Field(..., description="News sentiment: positive/negative/neutral")
    market_sentiment: Optional[str] = Field(None, description="Overall market sentiment")
    fear_greed_value: Optional[int] = Field(None, description="Fear & Greed index (0-100)")
    fear_greed_classification: Optional[str] = Field(None, description="Fear & Greed classification")
    
    # Investment recommendation
    investment_outlook: str = Field(..., description="Investment outlook: bullish/bearish/neutral") 
    risk_level: str = Field(..., description="Risk level: low/medium/high")
    recommendation_confidence: float = Field(..., description="Recommendation confidence 0.0-1.0")
    key_factors: List[str] = Field(default_factory=list, description="Key analysis factors")
    
    # Metadata
    coin_id: str = Field(..., description="CoinGecko coin ID")
    vs_currency: str = Field(..., description="Reference currency")
    timeframe: str = Field(..., description="Timeframe in days")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Timestamp")
    analysis_depth: str = Field(..., description="Analysis depth")
    execution_time_seconds: Optional[float] = Field(None, description="Execution time")
    api_calls_summary: Optional[Dict[str, Any]] = Field(None, description="API calls summary")


API_DESCRIPTION: str = (
    "AI-powered crypto analysis with autonomous tool selection and Fear & Greed Index integration."
)


def build_response(*, ok: bool, data: Dict[str, Any], errors: Optional[list] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": ok, "data": data or {}, "errors": errors or [], "meta": meta or {}}


class OutputCapture:
    """Captures stdout for tool call analysis"""
    def __init__(self):
        self.original_stdout = sys.stdout
        self.captured_text = ""
        
    def __enter__(self):
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        
    def write(self, text):
        self.captured_text += text
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
    def flush(self):
        self.original_stdout.flush()


def parse_tool_calls_from_output(output_text: str) -> Dict[str, Any]:
    """Parse debug logs to count API/function calls"""
    
    patterns = {
        "CoinGeckoToolKit": {
            "get_market_data": r"🎯 \[DEBUG\] get_market_data CHAMADA!",
            "get_coin_data": r"🎯 \[DEBUG\] get_coin_data CHAMADA!",
            "perform_technical_analysis": r"🎯 \[DEBUG\] perform_technical_analysis CHAMADA!",
            "get_coin_history": r"🎯 \[DEBUG\] get_coin_history CHAMADA!",
            "get_coin_chart": r"🎯 \[DEBUG\] get_coin_chart CHAMADA!",
        },
        "CoinDeskToolKit": {
            "get_latest_articles": r"🎯 \[DEBUG\] get_latest_articles CHAMADA!",
        },
        "FearGreedToolKit": {
            "get_current_fear_greed": r"📊 \*\*Fear and Greed Index\*\*",
        }
    }
    
    stats = {"total_calls": 0, "tools": {}}
    
    for tool_name, tool_patterns in patterns.items():
        tool_calls = {}
        total_tool_calls = 0
        
        for func_name, pattern in tool_patterns.items():
            count = len(re.findall(pattern, output_text))
            if count > 0:
                tool_calls[func_name] = count
                total_tool_calls += count
                stats["total_calls"] += count
        
        if total_tool_calls > 0:
            stats["tools"][tool_name] = {"functions": tool_calls, "total_calls": total_tool_calls}
    
    return stats


def run_analysis(coin_id: str, **params: Any) -> Dict[str, Any]:
    """Runs autonomous crypto analysis with tool call tracking"""
    
    start_time = time.time()
    request_id: str = str(uuid.uuid4())
    vs_currency: str = params.get("vs_currency") or os.getenv("DEFAULT_VS_CURRENCY", "usd")
    timeframe: str = params.get("timeframe", "30")
    analysis_depth: str = params.get("analysis_depth", "comprehensive")
    
    print(f"\n🚀 Starting {analysis_depth} crypto analysis...")
    print(f"📊 Asset: {coin_id.upper()}")
    print(f"💱 VS Currency: {vs_currency.upper()}")
    print(f"⏱️ Timeframe: {timeframe} days")
    print(f"🔢 Request ID: {request_id[:8]}...")
    print("─" * 60)
    
    try:
        print("🛠️ Initializing tools: CoinGecko, Fear&Greed, CoinDesk, Google, Reasoning...")
        
        agent = Agent(
            model=OpenRouter(id="openai/gpt-5-mini", api_key=os.getenv("OPENROUTER_API_KEY")),
            tools=[
                ReasoningTools(add_instructions=True),
                ThinkingTools(add_instructions=True), 
                GoogleSearchTools(),
                CoinGeckoToolKit(),
                CoinDeskToolKit(timeout=30),
                FearGreedToolKit(),
            ],
            response_model=CryptoAnalysis,
            description=(
                "Expert crypto analyst agent that autonomously chooses tools and creates structured analysis."
            ),
            instructions=[
                "MANDATORY WORKFLOW: 1) Get market data (CoinGecko), 2) Get Fear & Greed Index, 3) Technical analysis, 4) News sentiment, 5) Synthesize.",
                "ALWAYS use FearGreedToolKit.get_current_fear_greed() for sentiment data.",
                "Use CoinGecko for market data, technical analysis, and price levels.",
                "Use CoinDesk for news sentiment analysis.",
                "Fill fear_greed_value and fear_greed_classification with actual API data.",
                "Provide real resistance/support levels based on technical analysis.",
                "Include confidence scores and reasoning for all recommendations.",
            ],
            markdown=False,
        )
        
        print("🧠 Creating analysis prompt...")
        prompt = f"Analyze {coin_id} (vs_currency={vs_currency}, timeframe={timeframe}d, depth={analysis_depth}). Use FearGreedToolKit, CoinGecko, and CoinDesk tools."

        print("🤖 Agent starting analysis...")
        print("   📋 Monitoring tool calls...")
        
        with OutputCapture() as capture:
            agent_start = time.time()
            run_response = agent.run(prompt)
            agent_time = time.time() - agent_start
        
        print(f"\n✅ Agent completed in {agent_time:.2f}s")
        
        # Parse tool usage
        tool_stats = parse_tool_calls_from_output(capture.captured_text)
        
        print(f"\n📊 TOOL CALLS BREAKDOWN:")
        print(f"📞 Total API Calls: {tool_stats['total_calls']}")
        
        if tool_stats["tools"]:
            for tool_name, tool_data in tool_stats["tools"].items():
                print(f"🛠️ {tool_name}: {tool_data['total_calls']} calls")
                for func_name, count in tool_data["functions"].items():
                    print(f"   • {func_name}(): {count}x")
        else:
            print("   ℹ️ No API calls detected in debug logs")
        
        # Extract analysis data
        if hasattr(run_response, 'content') and isinstance(run_response.content, CryptoAnalysis):
            analysis_data = run_response.content.model_dump()
        elif hasattr(run_response, 'content'):
            analysis_data = run_response.content if isinstance(run_response.content, dict) else {}
        elif isinstance(run_response, CryptoAnalysis):
            analysis_data = run_response.model_dump()
        else:
            analysis_data = {"type": "failed_analysis", "summary": "Failed to extract analysis", "price_current": 0.0, "price_change_24h": 0.0, "price_trend": "unknown", "technical_signal": "hold", "resistance_levels": [], "support_levels": [], "news_sentiment": "neutral", "investment_outlook": "neutral", "risk_level": "high", "recommendation_confidence": 0.0, "key_factors": ["Analysis failed"]}
        
        total_time = time.time() - start_time
        
        analysis_data.update({
            "coin_id": coin_id,
            "vs_currency": vs_currency,
            "timeframe": timeframe,
            "analysis_depth": analysis_depth,
            "execution_time_seconds": round(total_time, 2),
            "api_calls_summary": tool_stats,
        })

        print(f"⏱️ Total time: {total_time:.2f}s")
        print("✅ Analysis completed!")
        print("─" * 60)

        return build_response(ok=True, data=analysis_data, errors=[], meta={"request_id": request_id})
        
    except Exception as exc:
        error_time = time.time() - start_time
        print(f"❌ Failed after {error_time:.2f}s: {str(exc)}")
        return build_response(ok=False, data={}, errors=[str(exc)], meta={"request_id": request_id})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=API_DESCRIPTION)
    parser.add_argument("coin_id", type=str, help="CoinGecko coin_id (e.g., bitcoin, ethereum)")
    parser.add_argument("--vs_currency", type=str, default="usd", help="Currency (default: usd)")
    parser.add_argument("--timeframe", type=str, default="30", help="Days (default: 30)")
    parser.add_argument("--analysis_depth", type=str, default="comprehensive", choices=["comprehensive", "quick", "technical_only"], help="Analysis depth")

    args = parser.parse_args()

    payload = run_analysis(coin_id=args.coin_id, vs_currency=args.vs_currency, timeframe=args.timeframe, analysis_depth=args.analysis_depth)
    print(json.dumps(payload, ensure_ascii=False, indent=2))