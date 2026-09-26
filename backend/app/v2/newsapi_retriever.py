import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Optional, Any, Dict
from backend.app.v2.schemas import (
    EvidenceItem,
    ExtractedClaim,
    EvidenceSourceType,
    StanceType
)
from backend.app.v2.fact_check_retriever import extract_domain_from_url
from backend.app.v2.query_builder import get_query_builder, FactCheckQueryBuilder


def load_env_key(var_name: str) -> str:
    """Loads an environment variable from os.environ or backend/.env securely."""
    val = os.environ.get(var_name, "").strip()
    if val:
        return val

    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.join(os.getcwd(), "backend", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == var_name:
                                return v.strip().strip("'\"")
            except Exception:
                pass
    return ""


class NewsAPIKeyError(ValueError):
    """Raised when NEWS_API_KEY is missing or empty in live mode."""
    pass


class NewsAPIError(RuntimeError):
    """Raised when NewsAPI returns an HTTP or network error status."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"NewsAPI error (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.message = message


class NewsAPIRateLimitError(NewsAPIError):
    """Raised when NewsAPI returns HTTP 429 rate limit status."""
    def __init__(self, message: str = "NewsAPI rate limit exceeded (HTTP 429)"):
        super().__init__(429, message)


class NewsAPIMalformedResponseError(ValueError):
    """Raised when NewsAPI returns malformed or non-parseable JSON response."""
    pass


class NewsAPIRetriever:
    """NewsAPI.org /v2/everything live-news retrieval adapter.

    Queries global news sources and maps article metadata into V2 EvidenceItem models.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock_mode: bool = False,
        timeout: float = 8.0,
        query_builder: Optional[FactCheckQueryBuilder] = None
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = load_env_key("NEWS_API_KEY")
        self.mock_mode = mock_mode
        self.timeout = timeout
        self.query_builder = query_builder or get_query_builder()
        self.api_call_count = 0

    def search_claim_news(
        self,
        claim: ExtractedClaim,
        max_results: int = 5
    ) -> List[EvidenceItem]:
        """Queries live news articles for an extracted claim with bounded results."""
        if not claim or not claim.text:
            return []

        self.api_call_count = 0

        if self.mock_mode:
            return self._get_mock_evidence(claim, max_results=max_results)

        if not self.api_key:
            raise NewsAPIKeyError(
                "NEWS_API_KEY environment variable is missing or empty. "
                "Set NEWS_API_KEY in backend/.env or enable mock_mode=True for offline testing."
            )

        query = self.query_builder.build_primary_query(claim)
        if not query:
            query = self.query_builder.build_fallback_query(claim)
        if not query:
            return []

        return self._execute_search_query(query, claim, max_results=max_results)

    def _execute_search_query(
        self,
        query: str,
        claim: ExtractedClaim,
        max_results: int = 5
    ) -> List[EvidenceItem]:
        """Executes a single HTTP search query against NewsAPI /v2/everything."""
        self.api_call_count += 1

        bounded_page_size = max(1, min(max_results, 5))
        params = {
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": bounded_page_size
        }
        request_url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        # Transmit API key via secure header rather than URL query parameter
        headers = {
            "X-Api-Key": self.api_key,
            "User-Agent": "RealTimeNewsVerificationSystem/2.0"
        }

        req = urllib.request.Request(request_url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as err:
                    raise NewsAPIMalformedResponseError(
                        f"Failed to parse JSON response from NewsAPI: {err}"
                    ) from err

                return self._normalize_response(data, claim)

        except urllib.error.HTTPError as http_err:
            error_body = ""
            try:
                error_body = http_err.read().decode("utf-8")
                err_json = json.loads(error_body)
                err_msg = err_json.get("message", http_err.reason)
            except Exception:
                err_msg = http_err.reason or error_body

            if http_err.code == 429:
                raise NewsAPIRateLimitError(err_msg) from http_err
            raise NewsAPIError(http_err.code, err_msg) from http_err

        except urllib.error.URLError as url_err:
            reason_str = str(url_err.reason)
            if "timed out" in reason_str.lower():
                raise NewsAPIError(504, f"NewsAPI gateway timeout: {reason_str}") from url_err
            raise NewsAPIError(503, f"NewsAPI network error: {reason_str}") from url_err

    def _normalize_response(self, data: Any, claim: ExtractedClaim) -> List[EvidenceItem]:
        """Normalizes raw NewsAPI JSON response payload into EvidenceItem list."""
        if not isinstance(data, dict):
            raise NewsAPIMalformedResponseError("Expected top-level JSON object in response.")

        api_status = data.get("status")
        if api_status == "error":
            code = data.get("code", "error")
            message = data.get("message", "NewsAPI returned error status.")
            if code == "rateLimited":
                raise NewsAPIRateLimitError(message)
            if code in ("apiKeyMissing", "apiKeyInvalid", "apiKeyDisabled"):
                raise NewsAPIError(401, message)
            raise NewsAPIError(400, message)

        articles_list = data.get("articles")
        if articles_list is None:
            return []

        if not isinstance(articles_list, list):
            raise NewsAPIMalformedResponseError("Field 'articles' must be a list if present.")

        evidence_items: List[EvidenceItem] = []

        for idx, article in enumerate(articles_list):
            if not isinstance(article, dict):
                continue

            source_info = article.get("source", {})
            if isinstance(source_info, dict):
                source_name = str(source_info.get("name", "Unknown Publisher")).strip() or "Unknown Publisher"
            else:
                source_name = "Unknown Publisher"

            title = str(article.get("title", "")).strip()
            # Filter out deleted/removed articles
            if not title or title.lower() == "[removed]":
                continue

            description = str(article.get("description", "")).strip()
            url = str(article.get("url", "")).strip()
            publish_date = article.get("publishedAt")

            domain = extract_domain_from_url(url, fallback_site=source_name)
            snippet = f"{description} (Source: {source_name})" if description else title

            item_id = f"news_{claim.claim_id}_{idx+1}"

            evidence_items.append(
                EvidenceItem(
                    id=item_id,
                    claim_id=claim.claim_id,
                    source_type=EvidenceSourceType.LIVE_NEWS_SEARCH,
                    publisher=source_name,
                    domain=domain,
                    url=url,
                    title=title,
                    snippet=snippet,
                    publish_date=publish_date,
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.NEUTRAL,
                    raw_rating=None,
                    claim_reviewed=None
                )
            )

        return evidence_items

    def _get_mock_evidence(self, claim: ExtractedClaim, max_results: int = 5) -> List[EvidenceItem]:
        """Generates deterministic mock evidence for testing without live network calls."""
        if not claim or not claim.text:
            return []

        c_id = claim.claim_id or "claim"
        claim_lower = claim.text.lower()

        # Generate realistic mock articles for known test entities
        if "nasa" in claim_lower or "mars" in claim_lower or "perseverance" in claim_lower:
            return [
                EvidenceItem(
                    id=f"news_mock_{c_id}_1",
                    claim_id=c_id,
                    source_type=EvidenceSourceType.LIVE_NEWS_SEARCH,
                    publisher="Reuters",
                    domain="reuters.com",
                    url=f"https://www.reuters.com/science/nasa-mission-mock-{c_id}",
                    title="NASA Mission Progress Reported",
                    snippet=f"NASA scientific teams continue mission operations. (Source: Reuters)",
                    publish_date="2026-09-25T12:00:00Z",
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.NEUTRAL,
                    raw_rating=None,
                    claim_reviewed=None
                )
            ][:max_results]

        if "federal reserve" in claim_lower or "interest rate" in claim_lower:
            return [
                EvidenceItem(
                    id=f"news_mock_{c_id}_1",
                    claim_id=c_id,
                    source_type=EvidenceSourceType.LIVE_NEWS_SEARCH,
                    publisher="Bloomberg",
                    domain="bloomberg.com",
                    url=f"https://www.bloomberg.com/news/fed-mock-{c_id}",
                    title="Federal Reserve Holds Policy Meeting",
                    snippet="Federal Reserve officials concluded monetary policy discussions. (Source: Bloomberg)",
                    publish_date="2026-09-25T14:00:00Z",
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.NEUTRAL,
                    raw_rating=None,
                    claim_reviewed=None
                )
            ][:max_results]

        return []


_newsapi_retriever_instance = None


def get_newsapi_retriever(api_key: Optional[str] = None, mock_mode: bool = False) -> NewsAPIRetriever:
    global _newsapi_retriever_instance
    if _newsapi_retriever_instance is None or mock_mode:
        _newsapi_retriever_instance = NewsAPIRetriever(api_key=api_key, mock_mode=mock_mode)
    return _newsapi_retriever_instance
