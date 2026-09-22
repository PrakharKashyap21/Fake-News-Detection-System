import os
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Optional, Dict, Any
from backend.app.v2.schemas import (
    EvidenceItem,
    ExtractedClaim,
    EvidenceSourceType,
    StanceType
)
from backend.app.v2.query_builder import get_query_builder, FactCheckQueryBuilder



class FactCheckAPIKeyError(ValueError):
    """Raised when GOOGLE_FACT_CHECK_API_KEY is missing in live mode."""
    pass


class FactCheckAPIError(RuntimeError):
    """Raised when Google Fact Check API returns an HTTP error status."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Google Fact Check API error (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.message = message


class FactCheckMalformedResponseError(ValueError):
    """Raised when Google Fact Check API returns malformed or non-parseable JSON response."""
    pass


def determine_stance(textual_rating: Optional[str]) -> StanceType:
    """Normalizes a raw textual rating string into a standard StanceType enum."""
    if not textual_rating:
        return StanceType.NEUTRAL

    rating_lower = str(textual_rating).strip().lower()

    # Contradicts / Debunked / False indicators
    false_keywords = [
        "false", "pants on fire", "pants-on-fire", "debunked", "incorrect",
        "inaccurate", "fake", "misleading", "untrue", "distorted", "scam",
        "hoax", "fiction", "mostly false", "four pinocchios", "4 pinocchios",
        "fake news", "fabricated", "disproven"
    ]
    for kw in false_keywords:
        if kw in rating_lower:
            return StanceType.CONTRADICTS

    # Supports / True / Accurate indicators
    true_keywords = [
        "true", "correct", "accurate", "verified", "mostly true",
        "supported", "authentic", "confirmed", "geppetto checkmark"
    ]
    for kw in true_keywords:
        if kw in rating_lower:
            return StanceType.SUPPORTS

    # Neutral / Mixed / Unproven / Half-True
    return StanceType.NEUTRAL


def extract_domain_from_url(url: str, fallback_site: Optional[str] = None) -> str:
    """Extracts clean hostname/domain from a review URL."""
    if url:
        try:
            parsed = urllib.parse.urlparse(url)
            netloc = parsed.netloc or parsed.path.split("/")[0]
            # Strip port and www.
            netloc = netloc.split(":")[0].lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            if netloc:
                return netloc
        except Exception:
            pass

    if fallback_site:
        site_clean = str(fallback_site).strip().lower()
        if site_clean.startswith("www."):
            site_clean = site_clean[4:]
        return site_clean

    return "unknown"


class GoogleFactCheckRetriever:

    """Adapter/Provider for querying Google Fact Check Tools Claim Search API

    and normalizing evidence into V2 EvidenceItem models with deterministic query refinement.
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock_mode: bool = False,
        query_builder: Optional[FactCheckQueryBuilder] = None
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "")
        self.mock_mode = mock_mode
        self.query_builder = query_builder or get_query_builder()
        self.api_call_count = 0  # Track API call count per claim for testing bounds

    def search_claim(self, claim: ExtractedClaim) -> List[EvidenceItem]:
        """Queries fact-check data for an extracted claim with bounded primary and fallback query attempts."""
        if not claim or not claim.text:
            return []

        self.api_call_count = 0

        if self.mock_mode:
            raw_mock = self._get_mock_evidence(claim)
            return self.query_builder.filter_relevant_evidence(claim, raw_mock)

        if not self.api_key:
            raise FactCheckAPIKeyError(
                "GOOGLE_FACT_CHECK_API_KEY environment variable is missing or empty. "
                "Set GOOGLE_FACT_CHECK_API_KEY or enable mock_mode=True for offline testing."
            )

        # Primary Query Attempt (Call 1)
        primary_query = self.query_builder.build_primary_query(claim)
        evidence_items = []

        if primary_query:
            evidence_items = self._execute_search_query(primary_query, claim)

        # Fallback Query Attempt (Call 2 - ONLY if Primary Query returned zero evidence)
        if not evidence_items:
            fallback_query = self.query_builder.build_fallback_query(claim)
            if fallback_query and fallback_query.lower() != primary_query.lower():
                evidence_items = self._execute_search_query(fallback_query, claim)

        # Lexical relevance filtering
        filtered_items = self.query_builder.filter_relevant_evidence(claim, evidence_items)
        return filtered_items

    def _execute_search_query(self, query: str, claim: ExtractedClaim) -> List[EvidenceItem]:
        """Executes a single HTTP search query against the Google Fact Check API."""
        if not query or not self.api_key:
            return []

        self.api_call_count += 1

        params = {
            "query": query,
            "key": self.api_key,
            "languageCode": "en"
        }
        request_url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(
            request_url,
            headers={"User-Agent": "RealTimeNewsVerificationSystem/2.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                body = response.read().decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as err:
                    raise FactCheckMalformedResponseError(
                        f"Failed to parse JSON response from Google Fact Check API: {err}"
                    ) from err

                return self._normalize_response(data, claim)

        except urllib.error.HTTPError as http_err:
            error_body = ""
            try:
                error_body = http_err.read().decode("utf-8")
            except Exception:
                pass
            raise FactCheckAPIError(http_err.code, error_body or http_err.reason) from http_err
        except urllib.error.URLError as url_err:
            raise FactCheckAPIError(503, f"Network/URL error: {url_err.reason}") from url_err


    def _normalize_response(self, data: Any, claim: ExtractedClaim) -> List[EvidenceItem]:
        """Normalizes raw Google API response payload into EvidenceItem list."""
        if not isinstance(data, dict):
            raise FactCheckMalformedResponseError("Expected top-level JSON object in response.")

        claims_list = data.get("claims")
        if claims_list is None:
            # Zero matching results is represented by absence of 'claims' key or empty list
            return []

        if not isinstance(claims_list, list):
            raise FactCheckMalformedResponseError("Field 'claims' must be a list if present.")

        evidence_items: List[EvidenceItem] = []

        for claim_idx, raw_claim in enumerate(claims_list):
            if not isinstance(raw_claim, dict):
                continue

            claim_reviews = raw_claim.get("claimReview", [])
            if not isinstance(claim_reviews, list):
                continue

            raw_claim_text = raw_claim.get("text", claim.text)
            claim_date = raw_claim.get("claimDate")

            for review_idx, review in enumerate(claim_reviews):
                if not isinstance(review, dict):
                    continue

                publisher_info = review.get("publisher", {})
                if not isinstance(publisher_info, dict):
                    publisher_info = {}

                publisher_name = str(publisher_info.get("name", "Unknown Publisher")).strip() or "Unknown Publisher"
                publisher_site = publisher_info.get("site")

                review_url = str(review.get("url", "")).strip()
                review_title = str(review.get("title", raw_claim_text)).strip() or raw_claim_text
                textual_rating = review.get("textualRating")
                review_date = review.get("reviewDate") or claim_date

                domain = extract_domain_from_url(review_url, fallback_site=publisher_site)
                stance = determine_stance(textual_rating)

                snippet = f"Reviewed Claim: '{raw_claim_text}' | Rating: {textual_rating or 'Unrated'}"

                item_id = f"fc_{claim.claim_id}_{claim_idx+1}_{review_idx+1}"

                evidence_items.append(
                    EvidenceItem(
                        id=item_id,
                        claim_id=claim.claim_id,
                        source_type=EvidenceSourceType.FACT_CHECK_API,
                        publisher=publisher_name,
                        domain=domain,
                        url=review_url,
                        title=review_title,
                        snippet=snippet,
                        publish_date=review_date,
                        credibility_score=None,
                        relevance_score=None,
                        stance=stance,
                        raw_rating=str(textual_rating) if textual_rating is not None else None
                    )
                )

        return evidence_items

    def _get_mock_evidence(self, claim: ExtractedClaim) -> List[EvidenceItem]:
        """Generates deterministic mock evidence for testing without live network calls."""
        claim_lower = claim.text.lower()

        # Generate mock refuting fact check if claim looks false/debunked
        if any(w in claim_lower for w in ["asteroid", "alien", "fake", "hoax", "secret"]):
            return [
                EvidenceItem(
                    id=f"fc_mock_{claim.claim_id}_1",
                    claim_id=claim.claim_id,
                    source_type=EvidenceSourceType.FACT_CHECK_API,
                    publisher="[MOCK] AP Fact Check",
                    domain="apnews.com",
                    url="https://apnews.com/article/fact-check-mock-debunk",
                    title="Mock Fact Check: Claim is false and lacks evidence",
                    snippet=f"Reviewed Claim: '{claim.text}' | Rating: False",
                    publish_date="2026-09-20T12:00:00Z",
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.CONTRADICTS,
                    raw_rating="False"
                ),
                EvidenceItem(
                    id=f"fc_mock_{claim.claim_id}_2",
                    claim_id=claim.claim_id,
                    source_type=EvidenceSourceType.FACT_CHECK_API,
                    publisher="[MOCK] PolitiFact",
                    domain="politifact.com",
                    url="https://www.politifact.com/factchecks/mock-pants-on-fire",
                    title="Mock Fact Check: Pants on Fire rating for claim",
                    snippet=f"Reviewed Claim: '{claim.text}' | Rating: Pants on Fire",
                    publish_date="2026-09-21T08:30:00Z",
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.CONTRADICTS,
                    raw_rating="Pants on Fire"
                )
            ]

        # Generate mock supporting fact check for verified factual claims
        if any(w in claim_lower for w in ["nasa", "launched", "federal reserve", "rates"]):
            return [
                EvidenceItem(
                    id=f"fc_mock_{claim.claim_id}_1",
                    claim_id=claim.claim_id,
                    source_type=EvidenceSourceType.FACT_CHECK_API,
                    publisher="[MOCK] FactCheck.org",
                    domain="factcheck.org",
                    url="https://www.factcheck.org/mock-verified-claim",
                    title="Mock Fact Check: Statement is accurate",
                    snippet=f"Reviewed Claim: '{claim.text}' | Rating: True",
                    publish_date="2026-09-21T10:00:00Z",
                    credibility_score=None,
                    relevance_score=None,
                    stance=StanceType.SUPPORTS,
                    raw_rating="True"
                )
            ]

        # Default empty search result for mock mode if claim doesn't trigger mock data
        return []


_fact_check_retriever_instance = None


def get_fact_check_retriever(api_key: Optional[str] = None, mock_mode: bool = False) -> GoogleFactCheckRetriever:
    global _fact_check_retriever_instance
    if _fact_check_retriever_instance is None or mock_mode:
        _fact_check_retriever_instance = GoogleFactCheckRetriever(api_key=api_key, mock_mode=mock_mode)
    return _fact_check_retriever_instance
