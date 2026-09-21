import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Optional, Any, Set
from backend.app.v2.schemas import (
    EvidenceItem,
    ExtractedClaim,
    EvidenceSourceType,
    StanceType
)
from backend.app.v2.fact_check_retriever import extract_domain_from_url


class NewsRetrieverAPIError(RuntimeError):
    """Raised when live news API returns an HTTP or network error status."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Live News API error (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.message = message


class NewsRetrieverMalformedResponseError(ValueError):
    """Raised when live news API returns malformed or non-parseable JSON response."""
    pass


class BaseNewsRetriever:
    """Base abstract provider interface for live news retrieval adapters."""

    def search_claim_news(
        self,
        claim: ExtractedClaim,
        max_results: int = 5,
        timespan: str = "24h"
    ) -> List[EvidenceItem]:
        raise NotImplementedError("Subclasses must implement search_claim_news method.")


def format_gdelt_seendate(seendate_raw: Optional[str]) -> Optional[str]:
    """Converts GDELT YYYYMMDDHHMMSS or YYYYMMDD string to ISO 8601 timestamp."""
    if not seendate_raw:
        return None
    clean_date = str(seendate_raw).strip()
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(?:T?(\d{2})(\d{2})(\d{2}))?", clean_date)
    if not match:
        return clean_date
    y, m, d, hh, mm, ss = match.groups()
    if hh is not None:
        return f"{y}-{m}-{d}T{hh}:{mm}:{ss}Z"
    return f"{y}-{m}-{d}T00:00:00Z"


class GDELTNewsRetriever(BaseNewsRetriever):
    """GDELT DOC 2.0 API live news retrieval adapter."""

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, mock_mode: bool = False, timeout: float = 5.0):
        self.mock_mode = mock_mode
        self.timeout = timeout

    def _construct_query(self, claim: ExtractedClaim) -> str:
        """Constructs an unquoted, intelligent search query optimized for GDELT recall."""
        if not claim:
            return ""

        # 1. Use keywords if available (unquoted, top 4 terms)
        if claim.keywords:
            clean_kws = []
            for kw in claim.keywords[:4]:
                k_str = re.sub(r'["\']', '', str(kw)).strip()
                if k_str and len(k_str) > 1:
                    clean_kws.append(k_str)
            if clean_kws:
                return " ".join(clean_kws)

        # 2. Fallback to clean claim text if keywords are missing
        if not claim.text:
            return ""

        raw_text = re.sub(r'["\']', '', str(claim.text)).strip()
        words = re.findall(r"\b[A-Za-z0-9-]+\b", raw_text)
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "who", "which", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "to", "from", "in", "out", "on", "off", "for",
            "with", "by", "at", "it", "its", "towards", "toward", "new"
        }
        content_words = [w for w in words if len(w) > 2 and w.lower() not in stopwords]
        if content_words:
            return " ".join(content_words[:4])

        return raw_text[:80]

    def _normalize_url(self, url: str) -> str:
        """Normalizes URL string for deduplication."""
        if not url:
            return ""
        clean_url = str(url).strip().lower()
        if clean_url.endswith("/"):
            clean_url = clean_url[:-1]
        return clean_url

    def search_claim_news(
        self,
        claim: ExtractedClaim,
        max_results: int = 5,
        timespan: str = "24h"
    ) -> List[EvidenceItem]:
        """Queries GDELT DOC API for current news reporting on an extracted claim."""
        if not claim or not claim.text:
            return []

        if self.mock_mode:
            return self._get_mock_news(claim, max_results=max_results, timespan=timespan)

        query = self._construct_query(claim)
        if not query:
            return []

        # Append sourcelang:english for GDELT API English language filtering
        gdelt_query = f"{query} sourcelang:english"

        params = {
            "query": gdelt_query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(max(1, max_results * 2), 250)),
            "timespan": str(timespan),
            "sort": "datedesc"
        }

        request_url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            request_url,
            headers={"User-Agent": "RealTimeNewsVerificationSystem/2.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as err:
                    raise NewsRetrieverMalformedResponseError(
                        f"Failed to parse JSON response from GDELT API: {err}"
                    ) from err

                return self._normalize_response(data, claim, max_results=max_results)

        except urllib.error.HTTPError as http_err:
            error_body = ""
            try:
                error_body = http_err.read().decode("utf-8")
            except Exception:
                pass
            raise NewsRetrieverAPIError(http_err.code, error_body or http_err.reason) from http_err
        except urllib.error.URLError as url_err:
            reason_str = str(url_err.reason)
            if "timed out" in reason_str.lower():
                raise NewsRetrieverAPIError(504, f"Request timed out after {self.timeout}s") from url_err
            raise NewsRetrieverAPIError(503, f"Network/URL error: {reason_str}") from url_err
        except TimeoutError as timeout_err:
            raise NewsRetrieverAPIError(504, f"Request timed out after {self.timeout}s") from timeout_err

    def _normalize_response(
        self,
        data: Any,
        claim: ExtractedClaim,
        max_results: int = 5
    ) -> List[EvidenceItem]:
        """Normalizes raw GDELT API JSON response into EvidenceItem objects with deduplication."""
        if not isinstance(data, dict):
            raise NewsRetrieverMalformedResponseError("Expected top-level JSON object in GDELT response.")

        articles_list = data.get("articles")
        if articles_list is None:
            # Empty search result
            return []

        if not isinstance(articles_list, list):
            raise NewsRetrieverMalformedResponseError("Field 'articles' must be a list if present.")

        evidence_items: List[EvidenceItem] = []
        seen_urls: Set[str] = set()

        for idx, article in enumerate(articles_list):
            if not isinstance(article, dict):
                continue

            raw_url = str(article.get("url", "")).strip()
            if not raw_url:
                continue

            norm_url = self._normalize_url(raw_url)
            if norm_url in seen_urls:
                # Deduplicate identical URLs
                continue
            seen_urls.add(norm_url)

            raw_title = str(article.get("title", "")).strip() or "Live News Article"
            raw_domain = article.get("domain")
            domain = extract_domain_from_url(raw_url, fallback_site=raw_domain)
            publisher_name = str(raw_domain).strip() if raw_domain else domain

            seendate_raw = article.get("seendate")
            publish_date = format_gdelt_seendate(seendate_raw)

            item_id = f"news_{claim.claim_id}_{len(evidence_items)+1}"
            snippet = f"Current News Report: '{raw_title}'"

            evidence_items.append(
                EvidenceItem(
                    id=item_id,
                    claim_id=claim.claim_id,
                    source_type=EvidenceSourceType.LIVE_NEWS_SEARCH,
                    publisher=publisher_name,
                    domain=domain,
                    url=raw_url,
                    title=raw_title,
                    snippet=snippet,
                    publish_date=publish_date,
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.NEUTRAL,
                    raw_rating=None
                )
            )

            if len(evidence_items) >= max_results:
                break

        return evidence_items

    def _get_mock_news(
        self,
        claim: ExtractedClaim,
        max_results: int = 5,
        timespan: str = "24h"
    ) -> List[EvidenceItem]:
        """Generates deterministic mock live news evidence for offline testing."""
        claim_lower = claim.text.lower() if claim and claim.text else ""

        mock_articles = [
            {
                "url": f"https://www.reuters.com/world/news-report-{claim.claim_id}-1",
                "title": f"Reuters Report: Current coverage on '{claim.text}'",
                "domain": "reuters.com",
                "seendate": "20260921T180000Z"
            },
            {
                "url": f"https://apnews.com/article/live-coverage-{claim.claim_id}-2",
                "title": f"Associated Press: Latest developments regarding claim",
                "domain": "apnews.com",
                "seendate": "20260921T191500Z"
            },
            {
                "url": f"https://www.bbc.com/news/world-report-{claim.claim_id}-3",
                "title": f"BBC News: Comprehensive report on ongoing events",
                "domain": "bbc.com",
                "seendate": "20260921T200000Z"
            }
        ]

        items: List[EvidenceItem] = []
        for idx, art in enumerate(mock_articles):
            if len(items) >= max_results:
                break
            items.append(
                EvidenceItem(
                    id=f"news_mock_{claim.claim_id}_{idx+1}",
                    claim_id=claim.claim_id,
                    source_type=EvidenceSourceType.LIVE_NEWS_SEARCH,
                    publisher=f"[MOCK] {art['domain']}",
                    domain=art["domain"],
                    url=art["url"],
                    title=art["title"],
                    snippet=f"Current News Report: '{art['title']}'",
                    publish_date=format_gdelt_seendate(art["seendate"]),
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.NEUTRAL,
                    raw_rating=None
                )
            )
        return items


_news_retriever_instance = None


def get_news_retriever(mock_mode: bool = False, timeout: float = 5.0) -> GDELTNewsRetriever:
    global _news_retriever_instance
    if _news_retriever_instance is None or mock_mode:
        _news_retriever_instance = GDELTNewsRetriever(mock_mode=mock_mode, timeout=timeout)
    return _news_retriever_instance
