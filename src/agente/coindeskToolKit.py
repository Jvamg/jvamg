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

        # Add API key to params instead of headers (omit when missing)
        if 'api_key' not in params and self.api_key:
            params['api_key'] = self.api_key

        # Set up headers according to CoinDesk API specification
        headers = {
            "User-Agent": "AGNO-CoinDesk-Toolkit/1.0",
            "Content-type": "application/json; charset=UTF-8",
            "Accept": "application/json"
        }

        # Make the request
        response = requests.get(
            url, params=params, headers=headers, timeout=self.timeout)

        print(f"📰 [DEBUG] Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ [DEBUG] Resposta de erro: {response.text}")

        response.raise_for_status()
        return response.json()

    def get_latest_articles(self, limit: int = 15, category: Optional[str] = None) -> str:
        """
        Get the latest news articles from CoinDesk.

        This function fetches the most recent articles from CoinDesk's API,
        providing up-to-date cryptocurrency and blockchain news.

        Args:
            limit (int): Number of articles to fetch (default is 15, max typically 50). Always clamped to at least 15 for reliable sentiment.
            category (Optional[str]): Filter articles by category (e.g., "BTC", "ETH", "MARKET", "REGULATION").

        Returns:
            str: A formatted string containing the latest articles information 
                 or an error message if the request fails.
        """
        print(
            f"🎯 [DEBUG] get_latest_articles CHAMADA! limit={limit}, category='{category}'")

        try:
            # Enforce minimum limit for meaningful sentiment analysis
            limit = 15 if limit is None else max(int(limit), 15)

            # Parameters for the CoinDesk API (match reference usage)
            params = {
                "lang": "EN",
                "limit": limit
            }
            # Use API's expected query key 'categories' only when filtering
            if category:
                params["categories"] = str(category).strip().upper()

            # Only apply category filtering when explicitly requested
            if category:
                print(
                    f"📝 [DEBUG] Categoria '{category}' será filtrada nos resultados (tickers recomendados como 'BTC', 'ETH')")

            try:
                print(f"🔄 [DEBUG] Chamando endpoint: news/v1/article/list")
                response_data = self._make_request(
                    "news/v1/article/list", params)
                print(f"✅ [DEBUG] Sucesso na chamada da API")

                # If category was specified, filter results by title/content after normalizing structure
                if category and response_data:
                    normalized = self._normalize_articles_payload(
                        response_data)
                    response_data = self._filter_articles_by_category(
                        normalized, category)

                # Debug: number of articles and oldest date
                try:
                    articles_list = self._articles_from_payload(response_data)
                    print(
                        f"📰 [DEBUG] Retrieved {len(articles_list)} article(s) from CoinDesk")
                    oldest_dt = self._oldest_article_datetime(articles_list)
                    if oldest_dt is not None:
                        print(
                            f"📰 [DEBUG] Oldest article date: {oldest_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    else:
                        print(f"📰 [DEBUG] Oldest article date: N/A")
                except Exception as dbg_e:
                    print(
                        f"⚠️ [DEBUG] Failed to compute article stats: {str(dbg_e)}")

            except requests.exceptions.RequestException as e:
                error_msg = f"Error fetching latest articles from CoinDesk: {str(e)}"
                print(f"❌ [DEBUG] Falha na API principal: {error_msg}")
                return error_msg

            formatted = self._format_latest_articles_response(
                response_data, limit)
            return formatted

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

        # Normalize payload once using robust extractor
        articles = self._articles_from_payload(articles_data)

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
            title = article.get('title', article.get(
                'TITLE', article.get('headline', 'N/A')))
            summary = article.get('summary', article.get(
                'BODY', article.get('excerpt', article.get('description', ''))))
            url = article.get('url', article.get(
                'URL', article.get('link', article.get('permalink', 'N/A'))))
            published_date = article.get('published_at', article.get(
                'PUBLISHED_ON', article.get('published', article.get('date', 'N/A'))))

            # Author can be a string or object in CoinDesk API
            author_data = article.get('author', article.get(
                'AUTHORS', article.get('by', {})))
            if isinstance(author_data, dict):
                author = author_data.get(
                    'name', author_data.get('display_name', 'CoinDesk'))
            else:
                author = str(author_data) if author_data else 'CoinDesk'

            # CoinDesk API may not provide sentiment, so we'll analyze it
            api_sentiment = article.get('sentiment', article.get(
                'SENTIMENT', self._analyze_sentiment(title, summary))).upper()
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
                        date_obj = datetime.fromisoformat(
                            str(published_date).replace('Z', '+00:00'))
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
            dominant_sentiment = max(
                sentiment_counts, key=sentiment_counts.get)
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
                        breakdown_parts.append(
                            f"{sentiment_label}: {count} ({pct}%)")
                result += ", ".join(breakdown_parts)

                # Add market insight
                if dominant_sentiment == "POSITIVE" and percentage > 60:
                    result += f"\n- **📊 Market Insight**: Strong positive sentiment suggests potential bullish momentum"
                elif dominant_sentiment == "NEGATIVE" and percentage > 60:
                    result += f"\n- **📊 Market Insight**: Strong negative sentiment indicates potential bearish pressure"
                elif dominant_sentiment == "NEUTRAL" or percentage < 40:
                    result += f"\n- **📊 Market Insight**: Mixed sentiment suggests market uncertainty or consolidation"

        print(
            f"✅ [DEBUG] Artigos formatados com sucesso - {len(articles)} artigos encontrados")
        return result

    def _normalize_articles_payload(self, payload: Dict) -> Dict | List:
        """
        Normalize CoinDesk payload to a predictable structure with an 'articles' list
        or return a plain list of article objects.
        """
        if not payload:
            return {"articles": []}
        if isinstance(payload, list):
            return payload
        if 'articles' in payload and isinstance(payload['articles'], list):
            return {"articles": payload['articles']}
        if 'data' in payload:
            data_field = payload['data']
            if isinstance(data_field, list):
                return data_field
            if isinstance(data_field, dict):
                # Direct 'articles' key
                if 'articles' in data_field and isinstance(data_field['articles'], list):
                    return {"articles": data_field['articles']}
                # Common alternatives: 'list', 'items', 'results', 'docs'
                for key in ("list", "items", "results", "docs"):
                    if key in data_field and isinstance(data_field[key], list):
                        return data_field[key]
                # Fallback: find first list-of-dicts anywhere one level deep
                for v in data_field.values():
                    if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                        return v
                # Nothing found; wrap
                return {"articles": [data_field]}
        if 'results' in payload and isinstance(payload['results'], list):
            return payload['results']
        # Try other common top-level keys
        for key in ("list", "items", "docs"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        # As a last resort, find first list-of-dicts among values
        for v in payload.values():
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                return v
        # Fallback: wrap as single article
        return {"articles": [payload]}

    def _articles_from_payload(self, payload: Dict | List) -> List[Dict]:
        """
        Return a flat list of article objects from any supported payload structure.
        """
        normalized = self._normalize_articles_payload(payload)
        if isinstance(normalized, list):
            return normalized
        return normalized.get('articles', []) or []

    def _oldest_article_datetime(self, articles: List[Dict]):
        """
        Compute the oldest published date from a list of article dicts.
        Supports numeric (seconds/ms) and ISO8601 strings.
        Fields checked: published_at, PUBLISHED_ON, published, date.
        Returns a timezone-aware datetime in UTC or None.
        """
        from datetime import datetime, timezone

        def to_dt(value):
            if value is None:
                return None
            try:
                # Numeric timestamp (seconds or milliseconds)
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                    iv = int(value)
                    # Heuristic: if milliseconds
                    if iv >= 10**12:
                        iv = iv // 1000
                    return datetime.fromtimestamp(iv, tz=timezone.utc)
                s = str(value)
                if 'T' in s:
                    # ISO8601, possibly with Z
                    return datetime.fromisoformat(s.replace('Z', '+00:00'))
                return None
            except Exception:
                return None

        datetimes = []
        for a in articles:
            raw = a.get('published_at', a.get(
                'PUBLISHED_ON', a.get('published', a.get('date'))))
            dt = to_dt(raw)
            if dt is not None:
                datetimes.append(dt)
        if not datetimes:
            return None
        return min(datetimes)

    def _analyze_sentiment(self, title: str, summary: str) -> str:
        """
        Simple sentiment analysis based on keywords since CoinDesk API may not provide sentiment.
        """
        text = f"{title} {summary}".lower()

        positive_words = ['bullish', 'surge', 'rally', 'gain', 'rise',
                          'growth', 'positive', 'adoption', 'breakthrough', 'success']
        negative_words = ['bearish', 'crash', 'fall', 'drop',
                          'decline', 'loss', 'negative', 'concern', 'risk', 'warning']

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
        Accepts synonyms and ticker symbols (e.g., 'bitcoin'/'btc', 'ethereum'/'eth').
        """
        if not articles_data or 'articles' not in articles_data:
            return articles_data

        category_lower = (category or '').strip().lower()
        synonym_map = {
            'bitcoin': {'bitcoin', 'btc'},
            'btc': {'bitcoin', 'btc'},
            'ethereum': {'ethereum', 'eth'},
            'eth': {'ethereum', 'eth'},
        }
        synonyms = set([category_lower]) | synonym_map.get(
            category_lower, set())

        filtered_articles = []

        def _flatten_strings(obj) -> List[str]:
            parts: List[str] = []
            try:
                if isinstance(obj, str):
                    parts.append(obj.lower())
                elif isinstance(obj, dict):
                    for v in obj.values():
                        parts.extend(_flatten_strings(v))
                elif isinstance(obj, (list, tuple, set)):
                    for v in obj:
                        parts.extend(_flatten_strings(v))
            except Exception:
                pass
            return parts
        for article in articles_data['articles']:
            combined = ' '.join(_flatten_strings(article))
            if any(s in combined for s in synonyms):
                filtered_articles.append(article)

        print(
            f"📊 [DEBUG] Filtrados {len(filtered_articles)} de {len(articles_data['articles'])} artigos para categoria '{category}'. Sinônimos usados: {sorted(list(synonyms))}")
        return {"articles": filtered_articles}
