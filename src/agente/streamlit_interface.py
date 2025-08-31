import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Crypto Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .success-alert {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .error-alert {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def check_api_health() -> bool:
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_available_coins() -> Dict[str, Any]:
    """Get list of available coins from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/coins", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"available_coins": []}
    except requests.exceptions.RequestException:
        return {"available_coins": []}

def analyze_crypto(coin_id: str, vs_currency: str, term_type: str) -> Dict[str, Any]:
    """Call the crypto analysis API"""
    try:
        payload = {
            "coin_id": coin_id,
            "vs_currency": vs_currency,
            "term_type": term_type
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"ok": False, "errors": [f"API returned status {response.status_code}"]}
            
    except requests.exceptions.RequestException as e:
        return {"ok": False, "errors": [f"Connection error: {str(e)}"]}

def display_analysis_results(data: Dict[str, Any]):
    """Display the analysis results in a structured format"""
    
    # Extract structured data sections
    obtainable = data.get('obtainable', {})
    thoughts = data.get('thoughts', {})
    metadata = data.get('metadata', {})
    
    # Header with coin info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"## 📊 {metadata.get('coin_id', 'N/A').upper()} Analysis")
        st.markdown(f"**Term:** {metadata.get('term_classification', 'N/A').upper()}")
    
    with col2:
        price = obtainable.get('price_current', 0)
        change_24h = obtainable.get('price_change_24h', 0)
        
        st.metric(
            label="Current Price",
            value=f"${price:,.2f}" if price else "N/A",
            delta=f"{change_24h:+.2f}%" if change_24h else None
        )
    
    with col3:
        st.metric(
            label="Technical Signal",
            value=thoughts.get('technical_signal', 'N/A').upper(),
            delta=None
        )
    
    # Summary
    st.markdown("### 📋 Executive Summary (AI Analysis)")
    st.markdown(f"**{thoughts.get('summary', 'No summary available')}**")
    
    # Data sections with clear labeling
    st.markdown("---")
    
    # Obtainable Data Section
    st.markdown("### 📊 Obtainable Data (From APIs)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Market Data")
        price = obtainable.get('price_current', 0)
        change = obtainable.get('price_change_24h', 0)
        volume = obtainable.get('volume_24h')
        market_cap = obtainable.get('market_cap')
        
        st.markdown(f"**Price:** ${price:,.2f}")
        change_color = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
        st.markdown(f"**24h Change:** {change_color} {change:+.2f}%")
        
        if volume:
            st.markdown(f"**24h Volume:** ${volume:,.0f}")
        if market_cap:
            st.markdown(f"**Market Cap:** ${market_cap:,.0f}")
        
    with col2:
        st.markdown("#### 📈 Technical Indicators")
        rsi = obtainable.get('rsi')
        if rsi:
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
            st.markdown(f"**RSI:** {rsi_color} {rsi:.1f} ({rsi_status})")
        
        sma20 = obtainable.get('sma_20')
        sma50 = obtainable.get('sma_50')
        sma200 = obtainable.get('sma_200')
        if sma20: st.markdown(f"**SMA 20:** ${sma20:,.2f}")
        if sma50: st.markdown(f"**SMA 50:** ${sma50:,.2f}")
        if sma200: st.markdown(f"**SMA 200:** ${sma200:,.2f}")
        
        macd = obtainable.get('macd_signal')
        if macd: st.markdown(f"**MACD:** {macd}")
    
    with col3:
        st.markdown("#### 😨 Fear & Greed Index")
        fg_value = obtainable.get('fear_greed_value')
        fg_class = obtainable.get('fear_greed_classification', 'N/A')
        
        if fg_value is not None:
            if fg_value <= 25:
                fg_color = "🔴"
            elif fg_value <= 45:
                fg_color = "🟠"
            elif fg_value <= 55:
                fg_color = "🟡"
            elif fg_value <= 75:
                fg_color = "🟢"
            else:
                fg_color = "🟢"
                
            st.markdown(f"{fg_color} **{fg_value}/100**")
            st.markdown(f"**{fg_class}**")
        else:
            st.markdown("**N/A**")
    
    # Thought Analysis Section
    st.markdown("### 🤖 AI Analysis (Thoughts)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 📈 Price Trend")
        trend = thoughts.get('price_trend', 'neutral')
        trend_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(trend.lower(), "🟡")
        st.markdown(f"{trend_color} **{trend.upper()}**")
    
    with col2:
        st.markdown("#### 🎯 Investment Outlook")
        outlook = thoughts.get('investment_outlook', 'neutral')
        outlook_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(outlook.lower(), "🟡")
        st.markdown(f"{outlook_color} **{outlook.upper()}**")
        
        risk = thoughts.get('risk_level', 'medium')
        risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk.lower(), "🟡")
        st.markdown(f"**Risk:** {risk_color} {risk.upper()}")
    
    with col3:
        st.markdown("#### 📰 Sentiment Analysis")
        news_sentiment = thoughts.get('news_sentiment', 'neutral')
        sent_color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(news_sentiment.lower(), "🟡")
        st.markdown(f"**News:** {sent_color} {news_sentiment.upper()}")
        
        market_sentiment = thoughts.get('market_sentiment')
        if market_sentiment:
            market_color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(market_sentiment.lower(), "🟡")
            st.markdown(f"**Market:** {market_color} {market_sentiment.upper()}")
    
    with col4:
        st.markdown("#### 🎯 AI Confidence")
        confidence = thoughts.get('recommendation_confidence', 0)
        confidence_pct = confidence * 100
        
        if confidence >= 0.8:
            conf_color = "🟢"
        elif confidence >= 0.6:
            conf_color = "🟡"
        else:
            conf_color = "🔴"
            
        st.markdown(f"{conf_color} **{confidence_pct:.0f}%**")
    
    # AI Support & Resistance Analysis
    st.markdown("#### 📈 AI-Identified Support & Resistance")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        resistance = thoughts.get('resistance_levels', [])
        if resistance:
            st.markdown("**AI Resistance Levels:**")
            for level in resistance[:3]:  # Show top 3
                st.markdown(f"• ${level:,.2f}")
        else:
            st.markdown("**AI Resistance Levels:** None identified")
    
    with tech_col2:
        support = thoughts.get('support_levels', [])
        if support:
            st.markdown("**AI Support Levels:**")
            for level in support[:3]:  # Show top 3
                st.markdown(f"• ${level:,.2f}")
        else:
            st.markdown("**AI Support Levels:** None identified")
    
    # AI Key Factors
    key_factors = thoughts.get('key_factors', [])
    if key_factors:
        st.markdown("### 🔑 AI Key Analysis Factors")
        for factor in key_factors:
            st.markdown(f"• {factor}")
    
    # Analysis Metadata
    with st.expander("📋 Analysis Metadata"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Analysis Type:** {data.get('type', 'N/A')}")
            st.markdown(f"**Coin ID:** {metadata.get('coin_id', 'N/A')}")
            st.markdown(f"**Range:** {metadata.get('range', 'N/A')} days")
            st.markdown(f"**Term Classification:** {metadata.get('term_classification', 'N/A').upper()}")
        
        with col2:
            st.markdown(f"**Execution Time:** {metadata.get('execution_time_seconds', 'N/A')} seconds")
            st.markdown(f"**Timestamp:** {metadata.get('timestamp', 'N/A')}")
            
            # API calls summary
            api_calls = metadata.get('api_calls_summary', {})
            if api_calls.get('total_calls', 0) > 0:
                st.markdown(f"**Total API Calls:** {api_calls['total_calls']}")
            
            # Token metrics
            token_metrics = metadata.get('token_metrics', {})
            if token_metrics:
                if isinstance(token_metrics, dict):
                    # Handle different token metric structures
                    total_tokens = 0
                    input_tokens = 0
                    output_tokens = 0
                    
                    # Try to get total_tokens (handle if it's a list)
                    total_raw = token_metrics.get('total_tokens')
                    if isinstance(total_raw, list) and total_raw:
                        total_tokens = sum(total_raw) if all(isinstance(x, (int, float)) for x in total_raw) else 0
                    elif isinstance(total_raw, (int, float)):
                        total_tokens = total_raw
                    
                    # Try to get input/output tokens
                    input_raw = token_metrics.get('input_tokens')
                    if isinstance(input_raw, list) and input_raw:
                        input_tokens = sum(input_raw) if all(isinstance(x, (int, float)) for x in input_raw) else 0
                    elif isinstance(input_raw, (int, float)):
                        input_tokens = input_raw
                        
                    output_raw = token_metrics.get('output_tokens')
                    if isinstance(output_raw, list) and output_raw:
                        output_tokens = sum(output_raw) if all(isinstance(x, (int, float)) for x in output_raw) else 0
                    elif isinstance(output_raw, (int, float)):
                        output_tokens = output_raw
                    
                    # If no total_tokens, calculate from input + output
                    if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
                        total_tokens = input_tokens + output_tokens
                    
                    # Display if we have valid data
                    if total_tokens > 0:
                        st.markdown(f"**Total Tokens:** {total_tokens:,}")
                        if input_tokens > 0:
                            st.markdown(f"  • Input: {input_tokens:,}")
                        if output_tokens > 0:
                            st.markdown(f"  • Output: {output_tokens:,}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Crypto Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.markdown('<div class="error-alert">❌ <strong>API Server Not Available</strong><br/>Please start the API server first:<br/><code>python src/agente/app.py serve</code></div>', unsafe_allow_html=True)
        st.stop()
    
    st.markdown('<div class="success-alert">✅ <strong>API Server Connected</strong></div>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.markdown("## ⚙️ Analysis Parameters")
    
    # Get available coins
    coins_data = get_available_coins()
    available_coins = coins_data.get('available_coins', [])
    
    # Create options for selectbox
    coin_options = {f"{coin['name']} ({coin['symbol']})": coin['id'] for coin in available_coins}
    coin_options["Custom Coin ID"] = "custom"
    
    # Coin selection
    selected_coin_display = st.sidebar.selectbox("Select Cryptocurrency", list(coin_options.keys()))
    
    if coin_options[selected_coin_display] == "custom":
        coin_id = st.sidebar.text_input("Enter Custom Coin ID", value="bitcoin", help="Use CoinGecko coin ID format")
    else:
        coin_id = coin_options[selected_coin_display]
    
    # Analysis parameters
    vs_currency = st.sidebar.selectbox("VS Currency", ["usd", "eur", "btc", "eth"], index=0)
    
    term_type = st.sidebar.selectbox(
        "Investment Term",
        ["short", "medium", "long"],
        index=0,
        help="Short: ~30 days, Medium: ~90 days, Long: ~365 days"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if analyze_button:
        if not coin_id:
            st.error("Please select or enter a valid coin ID")
            return
            
        # Show loading state
        with st.spinner(f"🔍 Analyzing {coin_id.upper()}... This may take a moment."):
            
            # Progress bar simulation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🛠️ Initializing tools...")
            progress_bar.progress(20)
            time.sleep(1)
            
            status_text.text("📊 Gathering market data...")
            progress_bar.progress(40)
            
            # Call the API
            start_time = time.time()
            result = analyze_crypto(coin_id, vs_currency, term_type)
            analysis_time = time.time() - start_time
            
            status_text.text("🧠 Processing AI analysis...")
            progress_bar.progress(80)
            time.sleep(0.5)
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Display results
        if result.get("ok"):
            st.success(f"✅ Analysis completed in {analysis_time:.2f} seconds")
            
            # Display the analysis
            analysis_data = result.get("data", {})
            display_analysis_results(analysis_data)
            
            # Raw data expander with structured sections
            with st.expander("🔍 Raw Analysis Data"):
                tab1, tab2, tab3 = st.tabs(["📊 Obtainable Data", "🤖 AI Thoughts", "⚙️ Metadata"])
                
                with tab1:
                    st.markdown("**Raw data obtained from APIs:**")
                    st.json(analysis_data.get('obtainable', {}))
                    
                with tab2:
                    st.markdown("**AI interpretations and analysis:**")
                    st.json(analysis_data.get('thoughts', {}))
                    
                with tab3:
                    st.markdown("**Analysis execution metadata:**")
                    st.json(analysis_data.get('metadata', {}))
                
        else:
            st.error("❌ Analysis failed")
            errors = result.get("errors", [])
            for error in errors:
                st.error(f"Error: {error}")
    
    else:
        # Welcome screen
        st.markdown("## 👋 Welcome to Crypto Analysis Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Features
            
            - **🤖 AI-Powered Analysis**: Advanced AI agent analyzes cryptocurrencies
            - **📊 Technical Indicators**: RSI, MACD, SMA 20/50/200 analysis
            - **💰 Market Data**: Price, volume, market cap from APIs
            - **😨 Fear & Greed Index**: Market sentiment integration
            - **📰 News Sentiment**: Latest news analysis
            - **🎯 Investment Terms**: Short, medium, and long-term strategies
            - **📈 Support/Resistance**: AI-identified price levels
            - **🔍 Clear Data Separation**: Obtainable facts vs AI interpretations
            
            ### 🚀 How to Use
            
            1. Select a cryptocurrency from the sidebar
            2. Choose your analysis parameters
            3. Click "Run Analysis" to get comprehensive insights
            4. Review obtainable data (facts) and AI thoughts (interpretations)
            5. Examine detailed analysis results with clear data separation
            """)
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            
            # Get some stats from available coins
            if available_coins:
                st.markdown(f"**Available Coins:** {len(available_coins)}")
                st.markdown("**Popular Options:**")
                for coin in available_coins[:5]:
                    st.markdown(f"• {coin['name']} ({coin['symbol']})")
                
                st.markdown("### 🔍 Data Structure")
                st.markdown("""
                **Obtainable Data:** Raw market data (price, volume, market cap), technical indicators (RSI, MACD, SMAs), Fear & Greed index
                
                **AI Thoughts:** Interpretations, recommendations, sentiment analysis
                
                **Metadata:** Analysis context and execution information
                """)
            
            st.markdown("### ⚙️ API Status")
            if api_healthy:
                st.markdown("🟢 **Connected**")
            else:
                st.markdown("🔴 **Disconnected**")

if __name__ == "__main__":
    main()
