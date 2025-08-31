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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
from agno.tools.googlesearch import GoogleSearchTools
from coingeckoToolKit import CoinGeckoToolKit
from coindeskToolKit import CoinDeskToolKit
from fearGreedToolKit import FearGreedToolKit

load_dotenv()

# FastAPI app initialization
app = FastAPI(
    title="Crypto Analysis API",
    description="AI-powered crypto analysis with autonomous tool selection and Fear & Greed Index integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class CryptoAnalysisRequest(BaseModel):
    """Request model for crypto analysis"""
    coin_id: str = Field(..., description="CoinGecko coin ID (e.g., bitcoin, ethereum)")
    vs_currency: str = Field(default="usd", description="Reference currency for API calls")
    term_type: str = Field(default="short", description="Term type: short, medium, long")

# Response models
class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    ok: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

# Structured models separating obtainable data from AI analysis
class ObtainableData(BaseModel):
    """Raw data obtained from APIs - objective facts"""
    
    # Market data (from CoinGecko API)
    price_current: float = Field(..., description="Current price from CoinGecko")
    price_change_24h: float = Field(..., description="24h price change % from CoinGecko")
    volume_24h: Optional[float] = Field(None, description="24h trading volume from CoinGecko")
    market_cap: Optional[float] = Field(None, description="Market capitalization from CoinGecko")
    
    # Technical indicators (calculated from CoinGecko data)
    rsi: Optional[float] = Field(None, description="RSI value (0-100) calculated")
    macd_signal: Optional[str] = Field(None, description="MACD signal calculated")
    sma_20: Optional[float] = Field(None, description="SMA 20 calculated")
    sma_50: Optional[float] = Field(None, description="SMA 50 calculated")
    sma_200: Optional[float] = Field(None, description="SMA 200 calculated")
    
    # Market sentiment indicators (from Fear & Greed API)
    fear_greed_value: Optional[int] = Field(None, description="Fear & Greed index (0-100) from Alternative.me")
    fear_greed_classification: Optional[str] = Field(None, description="Fear & Greed classification from API")


class ThoughtAnalysis(BaseModel):
    """AI interpretations and recommendations - subjective analysis"""
    
    # AI executive summary
    summary: str = Field(..., description="AI executive summary of findings")
    
    # AI interpretations of market data
    price_trend: str = Field(..., description="AI assessment: bullish/bearish/neutral")
    technical_signal: str = Field(..., description="AI recommendation: buy/sell/hold")
    
    # AI-calculated levels (based on technical analysis)
    resistance_levels: List[float] = Field(default_factory=list, description="AI-identified resistance levels")
    support_levels: List[float] = Field(default_factory=list, description="AI-identified support levels")
    
    # AI sentiment analysis
    news_sentiment: str = Field(..., description="AI assessment: positive/negative/neutral")
    market_sentiment: Optional[str] = Field(None, description="AI overall market assessment")
    
    # AI investment recommendations
    investment_outlook: str = Field(..., description="AI outlook: bullish/bearish/neutral")
    risk_level: str = Field(..., description="AI risk assessment: low/medium/high")
    recommendation_confidence: float = Field(..., description="AI confidence (0.0-1.0)")
    key_factors: List[str] = Field(default_factory=list, description="AI-identified key factors")


class AnalysisMetadata(BaseModel):
    """Analysis execution and context metadata"""
    
    # Request context
    coin_id: str = Field(..., description="CoinGecko coin ID")
    range: str = Field(..., description="Range in days")
    term_classification: str = Field(..., description="Term classification: short/medium/long")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Timestamp")
    
    # Execution metadata (added programmatically)
    execution_time_seconds: Optional[float] = Field(None, description="Execution time")
    api_calls_summary: Optional[Dict[str, Any]] = Field(None, description="API calls summary")
    token_metrics: Optional[Dict[str, Any]] = Field(None, description="Token usage metrics from agent execution")
    session_metrics: Optional[Dict[str, Any]] = Field(None, description="Session-level metrics from agent")


class CryptoAnalysis(BaseModel):
    """Complete cryptocurrency analysis with clear data separation"""
    
    type: str = Field(default="structured_crypto_analysis", description="Analysis type")
    
    # Structured data separation
    obtainable: ObtainableData = Field(..., description="Raw data obtained from APIs")
    thoughts: ThoughtAnalysis = Field(..., description="AI interpretations and recommendations")
    metadata: AnalysisMetadata = Field(..., description="Analysis execution metadata")


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


def make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format"""
    if obj is None:
        return None
    
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump()
        except Exception:
            pass
    
    # Handle objects with dict() method
    if hasattr(obj, 'dict'):
        try:
            return obj.dict()
        except Exception:
            pass
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle datetime objects
    if hasattr(obj, 'isoformat'):
        try:
            return obj.isoformat()
        except Exception:
            pass
    
    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Try to convert to dict for other objects
    try:
        if hasattr(obj, '__dict__'):
            return make_json_serializable(obj.__dict__)
        else:
            return str(obj)  # Fallback to string representation
    except Exception:
        return str(obj)


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
    
    # Get term type (default to "short")
    term_type = params.get("term_type", "short")
    term_classification = term_type.lower()
    if term_classification not in ["short", "medium", "long"]:
        term_classification = "short"  # fallback to short
    
    # Show term source
    term_source = "explicit" if params.get("term_type") else "default"
    
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
                max_tokens=12000  # Increased to prevent JSON truncation
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
            use_json_mode=True,  # Enabled to ensure consistent JSON output
            description=(
                "Expert crypto analyst agent that autonomously analyzes cryptocurrencies and returns structured data."
            ),
            instructions=[
                f"CRITICAL: You MUST return a complete CryptoAnalysis object with ALL 3 sections properly filled.",
                f"STRUCTURED OUTPUT: Follow the exact CryptoAnalysis schema with obtainable/thoughts/metadata sections.",
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
                f"7) Synthesize ALL data into the CryptoAnalysis structure with 3 sections:",
                f"   - obtainable: Raw data from APIs (prices, indicators, fear/greed)",
                f"   - thoughts: Your AI interpretations and recommendations", 
                f"   - metadata: Analysis context (coin_id, range, term_classification)",
                f"8) **SELF-REVIEW**: Review your analysis for consistency and term appropriateness",
                f"",
                f"FIELD REQUIREMENTS BY SECTION:",
                f"",
                f"OBTAINABLE DATA (from APIs) - RAW FACTS ONLY:",
                f"- price_current: Extract EXACT current price from CoinGeckoToolKit", 
                f"- price_change_24h: Extract EXACT 24h percentage change from CoinGeckoToolKit",
                f"- volume_24h: Extract EXACT 24h trading volume from CoinGeckoToolKit",
                f"- market_cap: Extract EXACT market capitalization from CoinGeckoToolKit",
                f"- rsi: RSI indicator calculated by CoinGeckoToolKit (number 0-100)",
                f"- macd_signal: MACD signal calculated by CoinGeckoToolKit (string)",
                f"- sma_20: SMA 20 calculated by CoinGeckoToolKit (number)",
                f"- sma_50: SMA 50 calculated by CoinGeckoToolKit (number)", 
                f"- sma_200: SMA 200 calculated by CoinGeckoToolKit (number)",
                f"- fear_greed_value: Use EXACT value from FearGreedToolKit (0-100 integer)",
                f"- fear_greed_classification: Use EXACT classification from FearGreedToolKit (string)",
                f"",
                f"THOUGHT ANALYSIS (your AI interpretations) - YOUR THINKING ONLY:",
                f"- summary: YOUR executive summary of complete analysis",
                f"- price_trend: YOUR interpretation - classify as 'bullish', 'bearish', or 'neutral'",
                f"- technical_signal: YOUR recommendation based on analysis - 'buy', 'sell', or 'hold'",
                f"- resistance_levels: YOUR identified resistance levels as array of prices",
                f"- support_levels: YOUR identified support levels as array of prices", 
                f"- news_sentiment: YOUR analysis of news - classify as 'positive', 'negative', or 'neutral'",
                f"- market_sentiment: YOUR overall market assessment",
                f"- investment_outlook: YOUR investment recommendation - 'bullish', 'bearish', or 'neutral'",
                f"- risk_level: YOUR risk assessment - classify as 'low', 'medium', or 'high'",
                f"- recommendation_confidence: YOUR confidence score (0.0 to 1.0)",
                f"- key_factors: YOUR list of 3-5 main factors that influenced your analysis",
                f"",
                f"ANALYSIS METADATA (context and execution info):",
                f"- coin_id: Set to '{coin_id}' (the coin being analyzed)",
                f"- range: Choose appropriate range for {term_classification} term (short~30d, medium~90d, long~365d)",
                f"- term_classification: Set to '{term_classification}' (investment time horizon)",
                f"- timestamp: Will be auto-generated",
                f"",
                f"CONSISTENCY VALIDATION RULES:",
                f"- Ensure thoughts.price_trend aligns with thoughts.technical_signal and thoughts.investment_outlook",
                f"- Verify obtainable.fear_greed_value matches obtainable.fear_greed_classification ranges",
                f"- Check that thoughts.risk_level is consistent with market volatility and sentiment",
                f"- Validate that thoughts.recommendation_confidence reflects the quality of available data",
                f"- Ensure thoughts.resistance_levels/support_levels are realistic based on obtainable.price_current",
                f"- Confirm thoughts.news_sentiment aligns with thoughts.market_sentiment",
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

        CRITICAL INSTRUCTIONS:
        You MUST adapt your analysis strategy and choose appropriate analysis range based on the {term_classification} term focus.
        
        IMPORTANT: You must structure your response with 3 clear sections:
        1. 'obtainable': Raw data from APIs (facts, numbers, classifications)
        2. 'thoughts': Your AI interpretations and recommendations  
        3. 'metadata': Analysis context and execution information
        
        FOR {term_classification.upper()} TERM ANALYSIS:
        {f"Focus on immediate price action, momentum indicators, volatility patterns, and day trading signals. Choose range around 30 days." if term_classification == "short" else 
         f"Balance technical and fundamental factors, swing trading opportunities, medium-term trends, and market cycles. Choose range around 90 days." if term_classification == "medium" else
         f"Emphasize fundamentals, macro trends, adoption metrics, long-term value proposition, and strategic investment thesis. Choose range around 365 days."}
        
        RETURN REQUIREMENTS:
        - Return a complete CryptoAnalysis object with 3 structured sections:
          * obtainable: Raw data from APIs (prices, indicators, fear/greed)
          * thoughts: Your AI interpretations and recommendations  
          * metadata: Analysis context (coin_id, range, term_classification)
        - Base your analysis specifically on {term_classification} term strategy
        - Ensure all data is real from your tool calls, not placeholder values
        
        CRITICAL DATA SEPARATION:
        - DO NOT interpret data in 'obtainable' - put EXACT values from APIs
        - DO ALL your thinking and interpretation in 'thoughts' section
        - Example: obtainable.fear_greed_value = 68 (exact), thoughts.market_sentiment = "bullish" (your interpretation)

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
        7) Fill ALL CryptoAnalysis sections with your {term_classification}-term focused findings:
           - obtainable: Put raw API data (price_current, rsi, fear_greed_value, etc.)
           - thoughts: Put your interpretations (summary, price_trend, technical_signal, etc.)
           - metadata: Put analysis context (coin_id, range, term_classification)
        8) **SELF-REVIEW STEP**: Before finalizing, carefully review your analysis:
           - Does thoughts.price_trend align with thoughts.technical_signal?
           - Is thoughts.investment_outlook consistent with all indicators?
           - Are thoughts.resistance_levels/support_levels realistic based on obtainable.price_current?
           - Does thoughts.recommendation_confidence reflect data quality and certainty?
           - Are all sentiment indicators (thoughts.news_sentiment, obtainable.fear_greed) logically consistent?
           - Do thoughts.key_factors actually explain your recommendation?
           - Did you choose an appropriate metadata.range for the {term_classification} term analysis?
           - Is your analysis specifically appropriate for {term_classification} term trading/investment?
           - Are you putting RAW DATA in obtainable and INTERPRETATIONS in thoughts?

        QUALITY ASSURANCE:
        Before finalizing, verify that your analysis makes logical sense as a whole.
        If any fields contradict each other, adjust them for consistency.
        Ensure all numeric values are realistic and all classifications are appropriate.
        Confirm that your recommendations and chosen range align with the {term_classification} term investment horizon.
        Only return the analysis when you're confident it's coherent and well-reasoned for {term_classification} term usage.

        Return the complete CryptoAnalysis object with all 3 sections properly structured:
        
        EXAMPLE STRUCTURE (follow this format exactly):
        {{
          "type": "structured_crypto_analysis",
          "obtainable": {{
            "price_current": [EXACT_PRICE_FROM_API],
            "price_change_24h": [EXACT_CHANGE_FROM_API],
            "volume_24h": [EXACT_VOLUME_FROM_API],
            "market_cap": [EXACT_MARKET_CAP_FROM_API],
            "rsi": [CALCULATED_RSI_NUMBER],
            "macd_signal": "[CALCULATED_MACD_SIGNAL]",
            "sma_20": [CALCULATED_SMA_NUMBER],
            "sma_50": [CALCULATED_SMA_NUMBER],
            "sma_200": [CALCULATED_SMA_NUMBER],
            "fear_greed_value": [EXACT_VALUE_FROM_API],
            "fear_greed_classification": "[EXACT_CLASSIFICATION_FROM_API]"
          }},
          "thoughts": {{
            "summary": "[YOUR_EXECUTIVE_SUMMARY]",
            "price_trend": "[YOUR_INTERPRETATION]",
            "technical_signal": "[YOUR_RECOMMENDATION]",
            "resistance_levels": [YOUR_IDENTIFIED_LEVELS],
            "support_levels": [YOUR_IDENTIFIED_LEVELS],
            "investment_outlook": "[YOUR_OUTLOOK]",
            "risk_level": "[YOUR_ASSESSMENT]",
            "recommendation_confidence": [YOUR_CONFIDENCE_0_TO_1]
          }},
          "metadata": {{
            "coin_id": "{coin_id}",
            "range": "[YOUR_CHOSEN_RANGE]",
            "term_classification": "{term_classification}"
          }}
        }}
        
        REMEMBER: obtainable = FACTS, thoughts = YOUR_ANALYSIS, metadata = CONTEXT
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
        
        # Extract CryptoAnalysis object with simplified parsing (JSON mode enabled)
        analysis_data = None
        
        try:
            if hasattr(run_response, 'content'):
                if isinstance(run_response.content, CryptoAnalysis):
                    print("✅ [DEBUG] Agent returned valid CryptoAnalysis object")
                    analysis_data = run_response.content.model_dump()
                elif isinstance(run_response.content, dict):
                        print("✅ [DEBUG] Agent returned dict, converting to CryptoAnalysis")
                        crypto_obj = CryptoAnalysis(**run_response.content)
                        analysis_data = crypto_obj.model_dump()
                elif isinstance(run_response.content, str):
                        print("🔧 [DEBUG] Agent returned string, attempting JSON parse...")
                        try:
                            json_data = json.loads(run_response.content)
                            crypto_obj = CryptoAnalysis(**json_data)
                            analysis_data = crypto_obj.model_dump()
                            print("✅ [DEBUG] Successfully parsed JSON string to CryptoAnalysis")
                        except json.JSONDecodeError as json_err:
                            print(f"❌ [DEBUG] JSON parse failed: {json_err}")
                            print(f"❌ [DEBUG] Raw content: {run_response.content[:500]}...")
                            raise Exception(f"Agent returned invalid JSON: {json_err}")
                else:
                        raise Exception(f"Unexpected content type: {type(run_response.content)}")
            elif isinstance(run_response, CryptoAnalysis):
                print("✅ [DEBUG] Direct CryptoAnalysis object received")
                analysis_data = run_response.model_dump()
            elif isinstance(run_response, dict):
                    print("✅ [DEBUG] Direct dict received, converting to CryptoAnalysis")
                    crypto_obj = CryptoAnalysis(**run_response)
                    analysis_data = crypto_obj.model_dump()
            elif isinstance(run_response, str):
                print("🔧 [DEBUG] Direct string received, attempting JSON parse...")
                try:
                    json_data = json.loads(run_response)
                    crypto_obj = CryptoAnalysis(**json_data)
                    analysis_data = crypto_obj.model_dump()
                    print("✅ [DEBUG] Successfully parsed direct JSON string to CryptoAnalysis")
                except json.JSONDecodeError as json_err:
                        print(f"❌ [DEBUG] JSON parse failed: {json_err}")
                        print(f"❌ [DEBUG] Raw response: {run_response[:500]}...")
                        raise Exception(f"Agent returned invalid JSON: {json_err}")
                else:
                    raise Exception(f"Unexpected response type: {type(run_response)}")
            
        except Exception as e:
            error_msg = f"Failed to extract CryptoAnalysis from agent response: {str(e)}"
            print(f"❌ [DEBUG] {error_msg}")
            raise Exception(error_msg)
        
        # Validate that agent actually filled critical fields with new structure
        critical_obtainable = ['price_current', 'price_change_24h']
        critical_thoughts = ['summary', 'technical_signal', 'investment_outlook', 'recommendation_confidence']
        critical_metadata = ['range', 'term_classification']
        
        missing_fields = []
        
        # Check obtainable data
        obtainable_data = analysis_data.get('obtainable', {})
        for field in critical_obtainable:
            if not obtainable_data.get(field):
                missing_fields.append(f"obtainable.{field}")
        
        # Check thought analysis
        thoughts_data = analysis_data.get('thoughts', {})
        for field in critical_thoughts:
            if not thoughts_data.get(field):
                missing_fields.append(f"thoughts.{field}")
        
        # Check metadata
        metadata_data = analysis_data.get('metadata', {})
        for field in critical_metadata:
            if not metadata_data.get(field):
                missing_fields.append(f"metadata.{field}")
        
        if missing_fields:
            raise Exception(f"Agent failed to fill critical fields: {missing_fields}")
        
        # Additional consistency validation (agent should have already done this)
        print("🔍 Performing final consistency checks...")
        
        # Check price trend vs technical signal alignment with new structure
        thoughts_data = analysis_data.get('thoughts', {})
        price_trend = thoughts_data.get('price_trend', '').lower()
        tech_signal = thoughts_data.get('technical_signal', '').lower()
        outlook = thoughts_data.get('investment_outlook', '').lower()
        
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
        confidence = thoughts_data.get('recommendation_confidence', 0)
        if confidence > 1.0 or confidence < 0.0:
            consistency_warnings.append(f"Recommendation confidence ({confidence}) outside valid range [0.0-1.0]")
        
        # Check term classification matches our calculation
        metadata_data = analysis_data.get('metadata', {})
        agent_term = metadata_data.get('term_classification', '').lower()
        expected_term = term_classification.lower()
        if agent_term != expected_term:
            consistency_warnings.append(f"Term classification mismatch: agent={agent_term}, expected={expected_term}")
        
        # Check if agent filled range field
        agent_range = metadata_data.get('range', '')
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
        
        # Extract token metrics from agent run response
        token_metrics = {}
        session_metrics = {}
        
        try:
            # Multiple ways to get token metrics from agno agent
            print("🔍 [DEBUG] Searching for token metrics...")
            
            # Method 1: run_response.metrics
            if hasattr(agent, 'run_response') and agent.run_response and hasattr(agent.run_response, 'metrics'):
                run_metrics_raw = agent.run_response.metrics
                if run_metrics_raw:
                    token_metrics = make_json_serializable(run_metrics_raw)
                    print(f"📊 [DEBUG] Found run_response.metrics: {token_metrics}")
            
            # Method 2: session_metrics
            if hasattr(agent, 'session_metrics') and agent.session_metrics:
                session_metrics_raw = agent.session_metrics
                if session_metrics_raw:
                    session_metrics = make_json_serializable(session_metrics_raw)
                    print(f"📊 [DEBUG] Found session_metrics: {session_metrics}")
            
            # Method 3: Check model metrics directly
            if hasattr(agent, 'model') and hasattr(agent.model, 'metrics'):
                model_metrics = agent.model.metrics
                if model_metrics:
                    if not token_metrics:  # Only use if we don't have run_response metrics
                        token_metrics = make_json_serializable(model_metrics)
                        print(f"📊 [DEBUG] Found model.metrics: {token_metrics}")
            
            # Method 4: Check if metrics are in the run_response itself
            if hasattr(run_response, 'metrics') and run_response.metrics:
                response_metrics = run_response.metrics
                if response_metrics and not token_metrics:
                    token_metrics = make_json_serializable(response_metrics)
                    print(f"📊 [DEBUG] Found response.metrics: {token_metrics}")
                    
            if not token_metrics and not session_metrics:
                print("⚠️ [DEBUG] No token metrics found in any location")
                
        except Exception as e:
            print(f"⚠️ [DEBUG] Error extracting metrics: {str(e)}")
            print(f"⚠️ [DEBUG] Available agent attributes: {[attr for attr in dir(agent) if not attr.startswith('_')]}")
            if hasattr(agent, 'run_response'):
                print(f"⚠️ [DEBUG] run_response attributes: {[attr for attr in dir(agent.run_response) if not attr.startswith('_')]}")
        
        # Agent already filled all fields - just add execution metadata
        if 'metadata' not in analysis_data:
            analysis_data['metadata'] = {}
            
        analysis_data['metadata'].update({
            "execution_time_seconds": round(total_time, 2),
            "api_calls_summary": tool_stats,
            "token_metrics": token_metrics if token_metrics else None,
            "session_metrics": session_metrics if session_metrics else None,
        })

        print(f"⏱️ Total time: {total_time:.2f}s")
        print("✅ Analysis completed!")
        print("─" * 60)

        return build_response(
            ok=True, 
            data=analysis_data, 
            errors=[], 
            meta={
                "request_id": request_id,
            }
        )
        
    except Exception as exc:
        error_time = time.time() - start_time
        print(f"❌ Failed after {error_time:.2f}s: {str(exc)}")
        return build_response(ok=False, data={}, errors=[str(exc)], meta={"request_id": request_id})


# FastAPI endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Crypto Analysis API",
        "description": API_DESCRIPTION,
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze - POST - Perform cryptocurrency analysis",
            "health": "/health - GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/analyze", response_model=ApiResponse)
async def analyze_crypto(request: CryptoAnalysisRequest):
    """
    Perform comprehensive cryptocurrency analysis
    
    Returns structured analysis with market data, technical indicators,
    sentiment analysis, and investment recommendations.
    """
    try:
        print(f"\n🌐 [FastAPI] New analysis request: {request.coin_id}")
        
        # Call the existing analysis function
        result = run_analysis(
            coin_id=request.coin_id,
            vs_currency=request.vs_currency,
            term_type=request.term_type
        )
        
        # Return the result directly (it's already in the correct format)
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ [FastAPI] Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/coins", response_model=Dict[str, Any])
async def get_available_coins():
    """Get list of popular cryptocurrencies available for analysis"""
    popular_coins = [
        {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
        {"id": "ethereum", "name": "Ethereum", "symbol": "ETH"},
        {"id": "binancecoin", "name": "BNB", "symbol": "BNB"},
        {"id": "cardano", "name": "Cardano", "symbol": "ADA"},
        {"id": "solana", "name": "Solana", "symbol": "SOL"},
        {"id": "ripple", "name": "XRP", "symbol": "XRP"},
        {"id": "polkadot", "name": "Polkadot", "symbol": "DOT"},
        {"id": "chainlink", "name": "Chainlink", "symbol": "LINK"},
        {"id": "litecoin", "name": "Litecoin", "symbol": "LTC"},
        {"id": "dogecoin", "name": "Dogecoin", "symbol": "DOGE"}
    ]
    
    return {
        "available_coins": popular_coins,
        "note": "You can use any valid CoinGecko coin ID, not just these popular ones"
    }

def start_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI server"""
    print(f"\n🚀 Starting Crypto Analysis API server...")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📖 API documentation will be available at: http://{host}:{port}/docs")
    print(f"📊 Alternative docs at: http://{host}:{port}/redoc")
    print("─" * 60)
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=API_DESCRIPTION)
    
    # Add subcommands for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # CLI analysis mode (original functionality)
    cli_parser = subparsers.add_parser("analyze", help="Run single analysis via CLI")
    cli_parser.add_argument("coin_id", type=str, help="CoinGecko coin_id (e.g., bitcoin, ethereum)")
    cli_parser.add_argument("--vs_currency", type=str, default="usd", help="Currency (default: usd)")
    cli_parser.add_argument("--term_type", type=str, default="short", choices=["short", "medium", "long"], help="Term type for analysis focus: short (~30d), medium (~90d), long (~365d). Default: short")
    
    # API server mode
    api_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    api_parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    api_parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")

    args = parser.parse_args()

    if args.mode == "serve":
        # Start FastAPI server
        start_api_server(host=args.host, port=args.port)
    elif args.mode == "analyze":
        # Original CLI functionality
        payload = run_analysis(coin_id=args.coin_id, vs_currency=args.vs_currency, term_type=args.term_type)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        # If no subcommand provided, show help
        parser.print_help()