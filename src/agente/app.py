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
    range: str = Field(..., description="Range in days")
    term_classification: str = Field(..., description="Term classification: short/medium/long")
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
    analysis_depth: str = params.get("analysis_depth", "comprehensive")
    
    # Get term type (default to "short")
    term_type = params.get("term_type", "short")
    term_classification = term_type.lower()
    if term_classification not in ["short", "medium", "long"]:
        term_classification = "short"  # fallback to short
    
    # Show term source
    term_source = "explicit" if params.get("term_type") else "default"
    
    print(f"\n🚀 Starting {analysis_depth} crypto analysis...")
    print(f"📊 Asset: {coin_id.upper()}")
    print(f"💱 VS Currency: {vs_currency.upper()}")
    print(f"🎯 Term Type: {term_classification.upper()} ({term_source})")
    print(f"📅 Agent will define analysis range based on {term_classification} term strategy")
    print(f"🔢 Request ID: {request_id[:8]}...")
    print("─" * 60)
    
    try:
        print("🛠️ Initializing tools: CoinGecko, Fear&Greed, CoinDesk, Google, Reasoning...")
        
        agent = Agent(
            model=OpenRouter(
                id="openai/gpt-5-mini", 
                api_key=os.getenv("OPENROUTER_API_KEY"),
                max_tokens=8000  # Further increase to prevent JSON truncation
            ),
            tools=[
                ReasoningTools(add_instructions=True),
                ThinkingTools(add_instructions=True), 
                GoogleSearchTools(),
                CoinGeckoToolKit(),
                CoinDeskToolKit(timeout=30),
                FearGreedToolKit(),
            ],
            response_model=CryptoAnalysis,
            # use_json_mode=True,  # ← Temporarily disabled to test structured outputs
            description=(
                "Expert crypto analyst agent that autonomously analyzes cryptocurrencies and returns structured data."
            ),
            instructions=[
                f"CRITICAL: You MUST return a complete CryptoAnalysis object with ALL fields properly filled.",
                f"STRUCTURED OUTPUT: Follow the exact CryptoAnalysis schema with all required fields.",
                f"",
                f"ANALYSIS STRATEGY FOR {term_classification.upper()} TERM:",
                f"- SHORT TERM: Focus on immediate price action, momentum, volatility, day trading signals (typically 30 days)",
                f"- MEDIUM TERM: Balance technical and fundamental analysis, swing trading opportunities (typically 90 days)",  
                f"- LONG TERM: Emphasize fundamentals, macro trends, long-term investment thesis (typically 365 days)",
                f"",
                f"REQUIRED ANALYSIS WORKFLOW:",
                f"1) Use CoinGeckoToolKit to get current market data for {coin_id}",
                f"2) Use FearGreedToolKit.get_current_fear_greed() for market sentiment",  
                f"3) Use CoinGeckoToolKit for technical analysis and price levels",
                f"4) Use CoinDeskToolKit for news sentiment",
                f"5) ADAPT your analysis focus based on the {term_classification} term strategy",
                f"6) CHOOSE appropriate range (days) for your {term_classification} term analysis",
                f"7) Synthesize ALL data into the CryptoAnalysis structure",
                f"8) **SELF-REVIEW**: Review your analysis for consistency and term appropriateness",
                f"",
                f"FIELD REQUIREMENTS:",
                f"- summary: Provide executive summary of your complete analysis",
                f"- price_current: Extract exact current price from market data", 
                f"- price_change_24h: Extract 24h percentage change from market data",
                f"- price_trend: Analyze and classify as 'bullish', 'bearish', or 'neutral'",
                f"- volume_24h: Include 24h trading volume from market data",
                f"- technical_signal: Based on technical analysis, return 'buy', 'sell', or 'hold'",
                f"- resistance_levels: Calculate actual resistance levels as array of prices",
                f"- support_levels: Calculate actual support levels as array of prices", 
                f"- news_sentiment: Analyze news and classify as 'positive', 'negative', or 'neutral'",
                f"- fear_greed_value: Use EXACT value from FearGreedToolKit (0-100 integer)",
                f"- fear_greed_classification: Use EXACT classification from FearGreedToolKit",
                f"- investment_outlook: Your investment recommendation: 'bullish', 'bearish', or 'neutral'",
                f"- risk_level: Assess and classify as 'low', 'medium', or 'high'",
                f"- recommendation_confidence: Your confidence score (0.0 to 1.0)",
                f"- key_factors: List 3-5 main factors that influenced your analysis",
                f"- range: Choose appropriate range for {term_classification} term (short~30d, medium~90d, long~365d)",
                f"- term_classification: Set to '{term_classification}' (investment time horizon)",
                f"",
                f"CONSISTENCY VALIDATION RULES:",
                f"- Ensure price_trend aligns with technical_signal and investment_outlook",
                f"- Verify fear_greed_value matches fear_greed_classification ranges",
                f"- Check that risk_level is consistent with market volatility and sentiment",
                f"- Validate that recommendation_confidence reflects the quality of available data",
                f"- Ensure resistance/support levels are realistic based on current price",
                f"- Confirm news_sentiment aligns with overall market_sentiment",
                f"",
                f"BEFORE FINALIZING: Review all fields for logical consistency and accuracy.",
                f"Use the tools to gather real-time data and base your analysis on actual market conditions.",
            ],
            markdown=False,
        )
        
        print("🧠 Creating analysis prompt...")
        prompt = f"""
        Perform a complete cryptocurrency analysis for {coin_id} specifically tailored for {term_classification.upper()} TERM trading/investment.

        Analysis Parameters:
        - Coin: {coin_id}
        - VS Currency: {vs_currency}
        - Term Classification: {term_classification} (REQUESTED ANALYSIS FOCUS)
        - Analysis Depth: {analysis_depth}

        CRITICAL INSTRUCTIONS:
        You MUST adapt your analysis strategy and choose appropriate analysis range based on the {term_classification} term focus:
        
        FOR {term_classification.upper()} TERM ANALYSIS:
        {f"Focus on immediate price action, momentum indicators, volatility patterns, and day trading signals. Choose range around 30 days." if term_classification == "short" else 
         f"Balance technical and fundamental factors, swing trading opportunities, medium-term trends, and market cycles. Choose range around 90 days." if term_classification == "medium" else
         f"Emphasize fundamentals, macro trends, adoption metrics, long-term value proposition, and strategic investment thesis. Choose range around 365 days."}
        
        RETURN REQUIREMENTS:
        - Return a complete CryptoAnalysis object with all fields filled
        - Base your analysis specifically on {term_classification} term strategy
        - Ensure all data is real from your tool calls, not placeholder values

        COMPLETE WORKFLOW:
        1) Get market data using CoinGeckoToolKit
        2) Get Fear & Greed Index using FearGreedToolKit
        3) Perform technical analysis using CoinGeckoToolKit (adjust focus for {term_classification} term)
        4) Get news sentiment using CoinDeskToolKit 
        5) DETERMINE appropriate range for {term_classification} term analysis:
           - SHORT term: Choose around 30 days for momentum/volatility focus
           - MEDIUM term: Choose around 90 days for trend analysis
           - LONG term: Choose around 365 days for fundamental analysis
        6) TAILOR your analysis specifically for {term_classification} term strategy
        7) Fill ALL CryptoAnalysis fields with your {term_classification}-term focused findings
        8) **SELF-REVIEW STEP**: Before finalizing, carefully review your analysis:
           - Does price_trend align with technical_signal?
           - Is investment_outlook consistent with all indicators?
           - Are resistance/support levels realistic based on current price?
           - Does recommendation_confidence reflect data quality and certainty?
           - Are all sentiment indicators (news_sentiment, fear_greed) logically consistent?
           - Do key_factors actually explain your recommendation?
           - Did you choose an appropriate range for the {term_classification} term analysis?
           - Is your analysis specifically appropriate for {term_classification} term trading/investment?

        QUALITY ASSURANCE:
        Before finalizing, verify that your analysis makes logical sense as a whole.
        If any fields contradict each other, adjust them for consistency.
        Ensure all numeric values are realistic and all classifications are appropriate.
        Confirm that your recommendations and chosen range align with the {term_classification} term investment horizon.
        Only return the analysis when you're confident it's coherent and well-reasoned for {term_classification} term usage.

        Return the complete CryptoAnalysis object with all fields populated based on your {term_classification}-term focused analysis.
        """

        print("🤖 Agent starting analysis...")
        print("   📋 Monitoring tool calls...")
        print("   🔍 Agent will self-review analysis for consistency...")
        
        with OutputCapture() as capture:
            agent_start = time.time()
            run_response = agent.run(prompt)
            agent_time = time.time() - agent_start
        
        print(f"\n✅ Agent completed in {agent_time:.2f}s")
        print("🔍 Verifying analysis consistency and quality...")
        
        # Debug: Check what the agent actually returned
        print(f"🔍 [DEBUG] Response type: {type(run_response)}")
        if hasattr(run_response, 'content'):
            print(f"🔍 [DEBUG] Content type: {type(run_response.content)}")
            if hasattr(run_response.content, 'model_dump'):
                print("🔍 [DEBUG] Content has model_dump method - appears to be CryptoAnalysis")
            else:
                print(f"🔍 [DEBUG] Content value: {str(run_response.content)[:200]}...")
        else:
            print(f"🔍 [DEBUG] No content attribute. Response: {str(run_response)[:200]}...")
        
        # Parse tool usage
        tool_stats = parse_tool_calls_from_output(capture.captured_text)
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"📞 Total API Calls: {tool_stats['total_calls']}")
        print(f"⏱️ Term Focus: {term_classification.upper()} term ({term_source})")
        print(f"🎯 Analysis Strategy: {'Day trading focused' if term_classification == 'short' else 'Swing trading balanced' if term_classification == 'medium' else 'Long-term investment focused'}")
        
        if tool_stats["tools"]:
            for tool_name, tool_data in tool_stats["tools"].items():
                print(f"🛠️ {tool_name}: {tool_data['total_calls']} calls")
                for func_name, count in tool_data["functions"].items():
                    print(f"   • {func_name}(): {count}x")
        else:
            print("   ℹ️ No API calls detected in debug logs")
        
        # Agent automatically fills all CryptoAnalysis fields after self-review
        total_time = time.time() - start_time
        
        # Try different ways to extract the CryptoAnalysis object
        analysis_data = None
        
        if hasattr(run_response, 'content'):
            if isinstance(run_response.content, CryptoAnalysis):
                print("✅ [DEBUG] Agent returned valid CryptoAnalysis object")
                analysis_data = run_response.content.model_dump()
            elif isinstance(run_response.content, dict):
                print("🔧 [DEBUG] Agent returned dict, checking if it's valid CryptoAnalysis data")
                try:
                    # Try to create CryptoAnalysis from dict
                    crypto_obj = CryptoAnalysis(**run_response.content)
                    analysis_data = crypto_obj.model_dump()
                    print("✅ [DEBUG] Successfully converted dict to CryptoAnalysis")
                except Exception as e:
                    print(f"❌ [DEBUG] Failed to convert dict to CryptoAnalysis: {e}")
            elif isinstance(run_response.content, str):
                print(f"⚠️ [DEBUG] Agent returned string: {run_response.content[:200]}...")
                print("🔧 [DEBUG] Attempting to parse string as JSON...")
                try:
                    import json
                    
                    # Try multiple JSON parsing strategies
                    json_content = run_response.content.strip()
                    
                    # Strategy 1: Direct parse
                    try:
                        parsed_json = json.loads(json_content)
                        crypto_obj = CryptoAnalysis(**parsed_json)
                        analysis_data = crypto_obj.model_dump()
                        print("✅ [DEBUG] Successfully parsed string as CryptoAnalysis (direct)")
                    except json.JSONDecodeError:
                        # Strategy 2: Find JSON block in markdown or mixed content
                        import re
                        json_pattern = r'\{[\s\S]*\}'
                        json_matches = re.findall(json_pattern, json_content)
                        
                        for match in json_matches:
                            try:
                                parsed_json = json.loads(match)
                                crypto_obj = CryptoAnalysis(**parsed_json)
                                analysis_data = crypto_obj.model_dump()
                                print("✅ [DEBUG] Successfully parsed JSON block as CryptoAnalysis")
                                break
                            except (json.JSONDecodeError, Exception):
                                continue
                        else:
                            # Strategy 3: Try to repair truncated JSON
                            print("🔧 [DEBUG] Attempting to repair truncated JSON...")
                            try:
                                # Add closing brackets if missing
                                if json_content.count('{') > json_content.count('}'):
                                    repaired = json_content + '}'
                                    parsed_json = json.loads(repaired)
                                    crypto_obj = CryptoAnalysis(**parsed_json)
                                    analysis_data = crypto_obj.model_dump()
                                    print("✅ [DEBUG] Successfully repaired and parsed truncated JSON")
                                else:
                                    raise json.JSONDecodeError("Could not repair JSON", json_content, 0)
                            except Exception:
                                raise json.JSONDecodeError("No valid JSON found", json_content, 0)
                            
                except json.JSONDecodeError as e:
                    print(f"❌ [DEBUG] JSON parsing failed: {e}")
                    print(f"🔍 [DEBUG] Problematic content: {run_response.content}")
                except Exception as e:
                    print(f"❌ [DEBUG] CryptoAnalysis creation from parsed JSON failed: {e}")
        elif isinstance(run_response, CryptoAnalysis):
            print("✅ [DEBUG] Direct CryptoAnalysis object received")
            analysis_data = run_response.model_dump()
        elif isinstance(run_response, dict):
            print("🔧 [DEBUG] Direct dict received, attempting conversion")
            try:
                crypto_obj = CryptoAnalysis(**run_response)
                analysis_data = crypto_obj.model_dump()
                print("✅ [DEBUG] Successfully converted dict to CryptoAnalysis")
            except Exception as e:
                print(f"❌ [DEBUG] Failed to convert dict to CryptoAnalysis: {e}")
        
        if analysis_data is None:
            # More detailed error message
            error_msg = f"Agent failed to return valid CryptoAnalysis structure. Response type: {type(run_response)}"
            if hasattr(run_response, 'content'):
                error_msg += f", Content type: {type(run_response.content)}"
            raise Exception(error_msg)
        
        # Validate that agent actually filled critical fields (including range defined by agent)
        critical_fields = ['summary', 'price_current', 'technical_signal', 'investment_outlook', 'recommendation_confidence', 'range', 'term_classification']
        missing_fields = [field for field in critical_fields if not analysis_data.get(field)]
        
        if missing_fields:
            raise Exception(f"Agent failed to fill critical fields: {missing_fields}")
        
        # Additional consistency validation (agent should have already done this)
        print("🔍 Performing final consistency checks...")
        
        # Check price trend vs technical signal alignment
        price_trend = analysis_data.get('price_trend', '').lower()
        tech_signal = analysis_data.get('technical_signal', '').lower()
        outlook = analysis_data.get('investment_outlook', '').lower()
        
        consistency_warnings = []
        
        if price_trend == 'bullish' and tech_signal == 'sell':
            consistency_warnings.append("Price trend (bullish) conflicts with technical signal (sell)")
        elif price_trend == 'bearish' and tech_signal == 'buy':
            consistency_warnings.append("Price trend (bearish) conflicts with technical signal (buy)")
            
        if outlook == 'bullish' and tech_signal == 'sell':
            consistency_warnings.append("Investment outlook (bullish) conflicts with technical signal (sell)")
        elif outlook == 'bearish' and tech_signal == 'buy':
            consistency_warnings.append("Investment outlook (bearish) conflicts with technical signal (buy)")
        
        # Check confidence score is reasonable
        confidence = analysis_data.get('recommendation_confidence', 0)
        if confidence > 1.0 or confidence < 0.0:
            consistency_warnings.append(f"Recommendation confidence ({confidence}) outside valid range [0.0-1.0]")
        
        # Check term classification matches our calculation
        agent_term = analysis_data.get('term_classification', '').lower()
        expected_term = term_classification.lower()
        if agent_term != expected_term:
            consistency_warnings.append(f"Term classification mismatch: agent={agent_term}, expected={expected_term}")
        
        # Check if agent filled range field
        agent_range = analysis_data.get('range', '')
        if not agent_range:
            consistency_warnings.append("Agent failed to define range field")
        
        # Validate range is reasonable for term type  
        if agent_range:
            try:
                range_int = int(agent_range)
                if term_classification == "short" and range_int > 60:
                    consistency_warnings.append(f"Range {agent_range} days seems too long for short term analysis")
                elif term_classification == "medium" and (range_int < 15 or range_int > 180):
                    consistency_warnings.append(f"Range {agent_range} days outside typical medium term range")
                elif term_classification == "long" and range_int < 90:
                    consistency_warnings.append(f"Range {agent_range} days seems too short for long term analysis")
            except ValueError:
                consistency_warnings.append(f"Range field '{agent_range}' is not a valid number")
        
        if consistency_warnings:
            print("⚠️ Consistency warnings found:")
            for warning in consistency_warnings:
                print(f"   • {warning}")
            print("ℹ️ Agent's self-review may need improvement for future analyses")
        else:
            print("✅ All consistency checks passed!")
            
        print("✅ Analysis quality validation completed!")
        
        # Agent already filled all fields - just add metadata
        analysis_data.update({
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
    parser.add_argument("--analysis_depth", type=str, default="comprehensive", choices=["comprehensive", "quick", "technical_only"], help="Analysis depth")
    parser.add_argument("--term_type", type=str, default="short", choices=["short", "medium", "long"], help="Term type for analysis focus: short (~30d), medium (~90d), long (~365d). Default: short")

    args = parser.parse_args()

    payload = run_analysis(coin_id=args.coin_id, vs_currency=args.vs_currency, analysis_depth=args.analysis_depth, term_type=args.term_type)
    print(json.dumps(payload, ensure_ascii=False, indent=2))