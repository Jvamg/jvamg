"""
Fear and Greed Index ToolKit for Agno Framework
===============================================

ToolKit para acessar o Fear and Greed Index da Alternative.me
API gratuita que fornece índice de medo e ganância do mercado crypto.
"""

import requests
from typing import Dict, Optional
from agno.tools import Toolkit


class FearGreedToolKit(Toolkit):
    """
    Fear and Greed Index ToolKit for crypto market sentiment analysis.
    Uses the free Alternative.me API to get market fear/greed sentiment data.
    
    API Documentation: https://alternative.me/crypto/fear-and-greed-index/
    """

    def __init__(self, timeout: Optional[int] = 10):
        super().__init__(name="fear_greed_index")
        self.timeout = timeout
        self.base_url = "https://api.alternative.me"
        
        # Register available tools
        self.register(self.get_current_fear_greed)
        self.register(self.get_fear_greed_history)

    def _make_request(self, endpoint_path: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Fear and Greed API"""
        try:
            url = f"{self.base_url}{endpoint_path}"
            headers = {
                "User-Agent": "FearGreedToolKit/1.0 (Crypto Analysis Agent)",
                "Accept": "application/json",
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except ValueError as e:
            return {"error": f"JSON decode failed: {str(e)}"}

    def get_current_fear_greed(self) -> str:
        """
        Get the current Fear and Greed Index value and classification.
        
        Returns:
            str: Formatted string with current fear/greed index data
        """
        try:
            data = self._make_request("/fng/")
            
            if "error" in data:
                return f"❌ Error fetching Fear and Greed Index: {data['error']}"
            
            if not data.get("data") or len(data["data"]) == 0:
                return "❌ No Fear and Greed Index data available"
            
            current_data = data["data"][0]
            value = current_data.get("value", "N/A")
            classification = current_data.get("value_classification", "Unknown")
            
            result = f"📊 **Fear and Greed Index**\n"
            result += f"Current Value: **{value}/100**\n"
            result += f"Classification: **{classification}**\n"
            
            # Add interpretation
            if classification.lower() == "extreme fear":
                result += "🔴 Market shows extreme fear - potential buying opportunity\n"
            elif classification.lower() == "fear":
                result += "🟠 Market shows fear - cautious sentiment\n"
            elif classification.lower() == "neutral":
                result += "🟡 Market sentiment is neutral - balanced emotions\n"
            elif classification.lower() == "greed":
                result += "🟢 Market shows greed - optimistic sentiment\n"
            elif classification.lower() == "extreme greed":
                result += "🔴 Market shows extreme greed - potential selling opportunity\n"
                
            result += f"\n📈 **Market Sentiment**: {classification} indicates {'selling pressure' if 'fear' in classification.lower() else 'buying pressure' if 'greed' in classification.lower() else 'balanced market'}."
            
            return result
            
        except Exception as e:
            return f"❌ Error processing Fear and Greed Index: {str(e)}"

    def get_fear_greed_history(self, limit: int = 30, date_format: str = "us") -> str:
        """
        Get historical Fear and Greed Index data.
        
        Args:
            limit (int): Number of days to retrieve (max 30, default 30)
            date_format (str): Date format - "us", "cn", "kr", "world" (default "us")
            
        Returns:
            str: Formatted string with historical fear/greed data
        """
        try:
            params = {
                "limit": min(limit, 30),  # API max is 30
                "format": date_format
            }
            
            data = self._make_request("/fng/", params=params)
            
            if "error" in data:
                return f"❌ Error fetching Fear and Greed history: {data['error']}"
            
            if not data.get("data") or len(data["data"]) == 0:
                return "❌ No historical Fear and Greed data available"
            
            history = data["data"]
            
            result = f"📊 **Fear and Greed Index - Last {len(history)} Days**\n\n"
            
            # Current value
            current = history[0]
            result += f"**Current**: {current.get('value', 'N/A')}/100 ({current.get('value_classification', 'Unknown')})\n\n"
            
            # Calculate trend
            if len(history) >= 7:
                recent_avg = sum(int(d.get('value', 0)) for d in history[:7]) / 7
                week_ago_avg = sum(int(d.get('value', 0)) for d in history[7:14]) / 7 if len(history) >= 14 else recent_avg
                
                trend = "increasing" if recent_avg > week_ago_avg else "decreasing" if recent_avg < week_ago_avg else "stable"
                result += f"**7-day trend**: {trend} (avg: {recent_avg:.1f} vs {week_ago_avg:.1f})\n\n"
            
            # Show distribution
            fear_count = sum(1 for d in history if int(d.get('value', 50)) < 25)
            neutral_count = sum(1 for d in history if 25 <= int(d.get('value', 50)) <= 75)
            greed_count = sum(1 for d in history if int(d.get('value', 50)) > 75)
            
            result += f"**Distribution ({limit} days)**:\n"
            result += f"- 🔴 Extreme Fear/Fear: {fear_count} days\n"
            result += f"- 🟡 Neutral: {neutral_count} days\n" 
            result += f"- 🟢 Greed/Extreme Greed: {greed_count} days\n\n"
            
            # Recent readings (last 5 days)
            result += "**Recent Values**:\n"
            for i, day in enumerate(history[:5]):
                value = day.get('value', 'N/A')
                classification = day.get('value_classification', 'Unknown')
                result += f"Day -{i}: {value}/100 ({classification})\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error processing Fear and Greed history: {str(e)}"
