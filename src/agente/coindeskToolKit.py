import requests
import os
from typing import Any, Dict, List, Optional
from agno.tools import Toolkit
from dotenv import load_dotenv
load_dotenv()


class CoinDeskToolKit(Toolkit):
    """
    CoinDesk Toolkit for fetching cryptocurrency news articles.
    This toolkit provides tools to get the latest articles and news 
    from CoinDesk through their API.

    Args:
        timeout (Optional[int]): Timeout for HTTP requests, default is 10 seconds.
    """

    def __init__(self, timeout: Optional[int] = 10):
        super().__init__(name="coindesk_tools")
        self.timeout: Optional[int] = timeout
        self.api_key = os.getenv("COINDESK_API_KEY")
        self.base_url = "https://data-api.coindesk.com"
        
        # Register the available tools
        self.register(self.get_latest_articles)

    def _make_request(self, endpoint_path: str, params: Optional[Dict] = None) -> Dict:
        """
        Generic method to make requests to CoinDesk API.
        
        Args:
            endpoint_path (str): The API endpoint path (e.g., "v1/articles/latest")
            params (Optional[Dict]): Query parameters for the request
        
        Returns:
            Dict: JSON response from the API
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        if params is None:
            params = {}
            
        # Check if API key is available
        if not self.api_key:
            print("⚠️ [DEBUG] API key não encontrada, usando key padrão")
        
        # Construct the full URL
        url = f"{self.base_url}/{endpoint_path}"
        
        # Add API key to params instead of headers
        if 'api_key' not in params:
            params['api_key'] = self.api_key
        
        # Set up headers according to CoinDesk API specification
        headers = {
            "User-Agent": "AGNO-CoinDesk-Toolkit/1.0",
            "Content-type": "application/json; charset=UTF-8"
        }
        
        # Make the request
        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        
        print(f"📰 [DEBUG] Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ [DEBUG] Resposta de erro: {response.text}")
        
        response.raise_for_status()
        return response.json()

    def get_latest_articles(self, limit: int = 10, category: Optional[str] = None) -> str:
        """
        Get the latest news articles from CoinDesk.

        This function fetches the most recent articles from CoinDesk's API,
        providing up-to-date cryptocurrency and blockchain news.

        Args:
            limit (int): Number of articles to fetch (default is 10, max typically 50).
            category (Optional[str]): Filter articles by category (e.g., "bitcoin", "ethereum", "markets").

        Returns:
            str: A formatted string containing the latest articles information 
                 or an error message if the request fails.
        """
        print(f"🎯 [DEBUG] get_latest_articles CHAMADA! limit={limit}, category='{category}'")

        try:
            # Parameters for the CoinDesk API
            params = {
                "lang": "EN",  # Language
                "limit": limit
            }
            
            # Note: category filtering may not be directly supported by this endpoint
            # We'll filter in post-processing if needed
            if category:
                print(f"📝 [DEBUG] Categoria '{category}' será filtrada nos resultados")

            try:
                print(f"🔄 [DEBUG] Chamando endpoint: news/v1/article/list")
                response_data = self._make_request("news/v1/article/list", params)
                print(f"✅ [DEBUG] Sucesso na chamada da API")
                
                # If category was specified, filter results by title/content
                if category and response_data:
                    response_data = self._filter_articles_by_category(response_data, category)
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Error fetching latest articles from CoinDesk: {str(e)}"
                print(f"❌ [DEBUG] Falha na API principal: {error_msg}")
                return error_msg

            return self._format_latest_articles_response(response_data, limit)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching latest articles from CoinDesk: {str(e)}"
            print(f"❌ [DEBUG] Erro de requisição: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error getting latest articles: {str(e)}"
            print(f"❌ [DEBUG] Erro inesperado: {error_msg}")
            return error_msg

    def _format_latest_articles_response(self, articles_data: Dict, limit: int) -> str:
        """
        Format the latest articles response into a readable string.
        """
        if not articles_data:
            no_data_msg = "No articles data found"
            print(f"❌ [DEBUG] {no_data_msg}")
            return no_data_msg

        # The response structure may vary, so we need to handle different formats
        articles = []
        
        # Try different possible response structures, prioritizing the expected format
        if isinstance(articles_data, list):
            articles = articles_data
        elif 'articles' in articles_data:
            articles = articles_data['articles']
        elif 'data' in articles_data:
            # CoinDesk API may return data in 'data' field
            data_field = articles_data['data']
            if isinstance(data_field, list):
                articles = data_field
            elif 'articles' in data_field:
                articles = data_field['articles']
            else:
                articles = [data_field]
        elif 'results' in articles_data:
            articles = articles_data['results']
        else:
            # If it's a single article object, wrap it in a list
            articles = [articles_data]

        if not articles or len(articles) == 0:
            no_articles_msg = "No articles found"
            print(f"❌ [DEBUG] {no_articles_msg}")
            return no_articles_msg

        # Format the response
        result = "📰 **Latest CoinDesk Articles**\n\n"
        
        # Track overall sentiment from API
        sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        
        for i, article in enumerate(articles[:limit], 1):
            # Extract article information with fallbacks for CoinDesk API response format
            title = article.get('title', article.get('TITLE', article.get('headline', 'N/A')))
            summary = article.get('summary', article.get('BODY', article.get('excerpt', article.get('description', ''))))
            url = article.get('url', article.get('URL', article.get('link', article.get('permalink', 'N/A'))))
            published_date = article.get('published_at', article.get('PUBLISHED_ON', article.get('published', article.get('date', 'N/A'))))
            
            # Author can be a string or object in CoinDesk API
            author_data = article.get('author', article.get('AUTHORS', article.get('by', {})))
            if isinstance(author_data, dict):
                author = author_data.get('name', author_data.get('display_name', 'CoinDesk'))
            else:
                author = str(author_data) if author_data else 'CoinDesk'
            
            # CoinDesk API may not provide sentiment, so we'll analyze it
            api_sentiment = article.get('sentiment', article.get('SENTIMENT', self._analyze_sentiment(title, summary))).upper()
            if api_sentiment in sentiment_counts:
                sentiment_counts[api_sentiment] += 1
            else:
                sentiment_counts['NEUTRAL'] += 1
            
            # Choose emoji based on API sentiment
            sentiment_emoji = {
                "POSITIVE": "📈 🟢",
                "NEUTRAL": "📊 ⚪",
                "NEGATIVE": "📉 🔴"
            }.get(api_sentiment, "📊 ⚪")
            
            # Format the published date if it's available
            if published_date and published_date != 'N/A':
                try:
                    from datetime import datetime
                    # Check if it's a Unix timestamp (integer)
                    if isinstance(published_date, (int, float)) or str(published_date).isdigit():
                        date_obj = datetime.fromtimestamp(int(published_date))
                        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                    # Try to parse as ISO format
                    elif 'T' in str(published_date):
                        date_obj = datetime.fromisoformat(str(published_date).replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                    else:
                        formatted_date = str(published_date)
                except:
                    formatted_date = str(published_date)
            else:
                formatted_date = "N/A"

            # Truncate summary if it's too long
            if summary and len(summary) > 200:
                summary = summary[:200] + "..."

            result += f"""**{i}. {title}**
👤 **Author**: {author}
📅 **Published**: {formatted_date}
{sentiment_emoji} **Sentiment**: {api_sentiment}
📝 **Summary**: {summary if summary else 'No summary available'}
🔗 **URL**: {url}

"""

        # Add overall sentiment summary
        result += f"📊 **Total Articles**: {len(articles[:limit])}\n\n"
        
        # Calculate overall market sentiment from API data
        total_analyzed = sum(sentiment_counts.values())
        if total_analyzed > 0:
            result += "🧠 **Overall Market Sentiment Analysis**:\n"
            
            # Find dominant sentiment
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            dominant_count = sentiment_counts[dominant_sentiment]
            
            if dominant_count > 0:
                percentage = round((dominant_count / total_analyzed) * 100, 1)
                
                sentiment_trend_emoji = {
                    "POSITIVE": "🚀 BULLISH",
                    "NEUTRAL": "📊 MIXED", 
                    "NEGATIVE": "🔻 BEARISH"
                }.get(dominant_sentiment, "📊 MIXED")
                
                result += f"- **Market Trend**: {sentiment_trend_emoji} ({percentage}% of articles show {dominant_sentiment.lower()} sentiment)\n"
                
                # Add breakdown
                result += "- **Sentiment Breakdown**: "
                breakdown_parts = []
                for sentiment, count in sentiment_counts.items():
                    if count > 0:
                        pct = round((count / total_analyzed) * 100, 1)
                        sentiment_label = {
                            "POSITIVE": "Positive",
                            "NEUTRAL": "Neutral", 
                            "NEGATIVE": "Negative"
                        }.get(sentiment, sentiment)
                        breakdown_parts.append(f"{sentiment_label}: {count} ({pct}%)")
                result += ", ".join(breakdown_parts)
                
                # Add market insight
                if dominant_sentiment == "POSITIVE" and percentage > 60:
                    result += f"\n- **📊 Market Insight**: Strong positive sentiment suggests potential bullish momentum"
                elif dominant_sentiment == "NEGATIVE" and percentage > 60:
                    result += f"\n- **📊 Market Insight**: Strong negative sentiment indicates potential bearish pressure"
                elif dominant_sentiment == "NEUTRAL" or percentage < 40:
                    result += f"\n- **📊 Market Insight**: Mixed sentiment suggests market uncertainty or consolidation"
        
        print(f"✅ [DEBUG] Artigos formatados com sucesso - {len(articles)} artigos encontrados")
        return result



    def _analyze_sentiment(self, title: str, summary: str) -> str:
        """
        Simple sentiment analysis based on keywords since CoinDesk API may not provide sentiment.
        """
        text = f"{title} {summary}".lower()
        
        positive_words = ['bullish', 'surge', 'rally', 'gain', 'rise', 'growth', 'positive', 'adoption', 'breakthrough', 'success']
        negative_words = ['bearish', 'crash', 'fall', 'drop', 'decline', 'loss', 'negative', 'concern', 'risk', 'warning']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'POSITIVE'
        elif negative_count > positive_count:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'

    def _filter_articles_by_category(self, articles_data: Dict, category: str) -> Dict:
        """
        Filter articles by category since the API may not support direct filtering.
        """
        if not articles_data or 'articles' not in articles_data:
            return articles_data
        
        category_lower = category.lower()
        filtered_articles = []
        
        for article in articles_data['articles']:
            # Check title and summary for category keywords
            title = str(article.get('title', '')).lower()
            summary = str(article.get('summary', '')).lower()
            
            if (category_lower in title or 
                category_lower in summary or
                (category_lower == 'bitcoin' and ('btc' in title or 'bitcoin' in title)) or
                (category_lower == 'ethereum' and ('eth' in title or 'ethereum' in title))):
                filtered_articles.append(article)
        
        print(f"📊 [DEBUG] Filtrados {len(filtered_articles)} de {len(articles_data['articles'])} artigos para categoria '{category}'")
        return {"articles": filtered_articles}
