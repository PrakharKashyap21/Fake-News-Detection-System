import json
import pytest
from unittest.mock import MagicMock, patch
import urllib.error

from backend.app.v2.schemas import (
    ExtractedClaim,
    EvidenceItem,
    EvidenceSourceType,
    StanceType,
    ClaimVerdict,
    OverallAssessment,
    VerificationRequest
)
from backend.app.v2.news_retriever import (
    GDELTNewsRetriever,
    NewsRetrieverAPIError,
    NewsRetrieverRateLimitError,
    NewsRetrieverMalformedResponseError
)
from backend.app.v2.fact_check_retriever import GoogleFactCheckRetriever
from backend.app.v2.verification_service import VerificationService


def test_successful_gdelt_response():
    """Tests normal successful parsing of GDELT DOC 2.0 JSON response."""
    mock_claim = ExtractedClaim(claim_id="c1", text="NASA Perseverance Rover Mars", keywords=["NASA", "Perseverance", "Mars"])
    mock_json_payload = {
        "articles": [
            {
                "url": "https://www.reuters.com/science/nasa-perseverance-mars-123",
                "title": "NASA Rover Discovers Organic Compounds on Mars",
                "domain": "reuters.com",
                "seendate": "20260925T120000Z"
            },
            {
                "url": "https://www.bbc.com/news/science-mars-rover-456",
                "title": "Perseverance Finds Carbon-Based Molecules",
                "domain": "bbc.com",
                "seendate": "20260925T140000Z"
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(mock_json_payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    retriever = GDELTNewsRetriever(mock_mode=False, timeout=15.0)

    with patch("urllib.request.urlopen", return_value=mock_response):
        items = retriever.search_claim_news(mock_claim, max_results=5)

    assert len(items) == 2
    assert items[0].domain == "reuters.com"
    assert items[0].source_type == EvidenceSourceType.LIVE_NEWS_SEARCH
    assert items[0].stance == StanceType.NEUTRAL
    assert items[1].domain == "bbc.com"


def test_http_429_rate_limit_error():
    """Tests recognition of HTTP 429 rate limit response raising NewsRetrieverRateLimitError."""
    mock_claim = ExtractedClaim(claim_id="c1", text="NASA")
    retriever = GDELTNewsRetriever(mock_mode=False, timeout=15.0)

    http_429_err = urllib.error.HTTPError(
        url="https://api.gdeltproject.org/api/v2/doc/doc",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=MagicMock(read=MagicMock(return_value=b"Please limit requests to one every 5 seconds"))
    )

    with patch("urllib.request.urlopen", side_effect=http_429_err):
        with pytest.raises(NewsRetrieverRateLimitError) as exc_info:
            retriever.search_claim_news(mock_claim)

    assert exc_info.value.status_code == 429
    assert "Please limit requests" in exc_info.value.message or "429" in str(exc_info.value)


def test_timeout_error():
    """Tests timeout error raising NewsRetrieverAPIError with status 504."""
    mock_claim = ExtractedClaim(claim_id="c1", text="Federal Reserve")
    retriever = GDELTNewsRetriever(mock_mode=False, timeout=15.0)

    timeout_err = urllib.error.URLError(reason="timed out")

    with patch("urllib.request.urlopen", side_effect=timeout_err):
        with pytest.raises(NewsRetrieverAPIError) as exc_info:
            retriever.search_claim_news(mock_claim)

    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.message.lower()


def test_malformed_response():
    """Tests handling of invalid non-JSON payload raising NewsRetrieverMalformedResponseError."""
    mock_claim = ExtractedClaim(claim_id="c1", text="Climate Summit")
    retriever = GDELTNewsRetriever(mock_mode=False, timeout=15.0)

    mock_response = MagicMock()
    mock_response.read.return_value = b"<html>502 Bad Gateway Server Error</html>"
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(NewsRetrieverMalformedResponseError):
            retriever.search_claim_news(mock_claim)


def test_empty_results_response():
    """Tests handling of empty articles array returning empty list."""
    mock_claim = ExtractedClaim(claim_id="c1", text="Obscure Local Bakery Sourdough")
    retriever = GDELTNewsRetriever(mock_mode=False, timeout=15.0)

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"articles": []}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        items = retriever.search_claim_news(mock_claim)

    assert isinstance(items, list)
    assert len(items) == 0


def test_verification_continuing_when_gdelt_fails_but_fact_check_exists():
    """
    Tests that when GDELT returns HTTP 429 rate-limited error,
    the pipeline records live_news_api = 'rate_limited', returns empty live news evidence,
    AND continues using Google Fact Check evidence to reach a valid claim verdict.
    """
    mock_claim = ExtractedClaim(
        claim_id="claim_fc",
        text="James Webb Telescope discovered deepest infrared image",
        keywords=["James Webb", "Telescope", "Infrared"]
    )

    mock_fc_evidence = [
        EvidenceItem(
            id="fc_1",
            claim_id="claim_fc",
            source_type=EvidenceSourceType.FACT_CHECK_API,
            publisher="FactCheck.org",
            domain="factcheck.org",
            url="https://factcheck.org/webb-image",
            title="Fact Check: Statement is verified true",
            snippet="Reviewed Claim: James Webb Telescope discovered deepest infrared image | Rating: True",
            stance=StanceType.SUPPORTS,
            raw_rating="True"
        )
    ]

    mock_extractor = MagicMock()
    mock_extractor.extract_claims.return_value = [mock_claim]

    mock_fc_retriever = MagicMock()
    mock_fc_retriever.search_claim.return_value = mock_fc_evidence

    mock_news_retriever = MagicMock()
    mock_news_retriever.search_claim_news.side_effect = NewsRetrieverRateLimitError("Rate limit exceeded")

    service = VerificationService(
        claim_extractor=mock_extractor,
        fc_retriever=mock_fc_retriever,
        news_retriever=mock_news_retriever,
        mock_mode=False
    )

    request = VerificationRequest(
        title="Webb Space Telescope Landmark Discovery",
        text="James Webb Telescope discovered deepest infrared image",
        include_linguistic_signal=False
    )

    response = service.verify_news(request)

    # 1. Verify service status
    assert response.service_status["fact_check_api"] == "ok"
    assert response.service_status["live_news_api"] == "rate_limited"

    # 2. Verify evidence was preserved and verdict reached
    assert len(response.claims) == 1
    c_detail = response.claims[0]
    assert c_detail.verdict == ClaimVerdict.SUPPORTED
    assert response.overall_assessment == OverallAssessment.SUPPORTED
    assert c_detail.supporting_evidence_count == 1
