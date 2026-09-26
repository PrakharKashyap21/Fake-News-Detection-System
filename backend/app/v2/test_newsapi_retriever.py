import json
import pytest
import urllib.error
from unittest.mock import MagicMock, patch

from backend.app.v2.schemas import (
    ExtractedClaim,
    EvidenceItem,
    EvidenceSourceType,
    StanceType,
    ClaimVerdict,
    OverallAssessment,
    VerificationRequest
)
from backend.app.v2.newsapi_retriever import (
    NewsAPIRetriever,
    NewsAPIKeyError,
    NewsAPIError,
    NewsAPIRateLimitError,
    NewsAPIMalformedResponseError
)
from backend.app.v2.verification_service import VerificationService


def test_successful_newsapi_parsing_and_metadata_mapping():
    """Tests normal successful parsing of NewsAPI /v2/everything JSON payload."""
    claim = ExtractedClaim(claim_id="c1", text="NASA Perseverance Rover Mars", keywords=["NASA", "Perseverance", "Mars"])
    mock_payload = {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": "reuters", "name": "Reuters"},
                "author": "Jane Doe",
                "title": "NASA Rover Discovers Organic Molecules on Mars",
                "description": "Perseverance rover collected rock samples containing organic carbon.",
                "url": "https://www.reuters.com/science/nasa-perseverance-123",
                "urlToImage": "https://www.reuters.com/image.jpg",
                "publishedAt": "2026-09-25T12:00:00Z",
                "content": "Full article content here..."
            },
            {
                "source": {"id": None, "name": "BBC News"},
                "author": None,
                "title": "Mars Mission Update",
                "description": "Scientists report new findings from Jezero Crater.",
                "url": "https://www.bbc.com/news/mars-456",
                "urlToImage": None,
                "publishedAt": "2026-09-25T14:00:00Z",
                "content": None
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    retriever = NewsAPIRetriever(api_key="mock_key", mock_mode=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        items = retriever.search_claim_news(claim, max_results=5)

    assert len(items) == 2
    # Check item 1 mapping
    assert items[0].id == "news_c1_1"
    assert items[0].claim_id == "c1"
    assert items[0].source_type == EvidenceSourceType.LIVE_NEWS_SEARCH
    assert items[0].publisher == "Reuters"
    assert items[0].domain == "reuters.com"
    assert items[0].url == "https://www.reuters.com/science/nasa-perseverance-123"
    assert items[0].title == "NASA Rover Discovers Organic Molecules on Mars"
    assert "Perseverance rover collected rock samples" in items[0].snippet
    assert items[0].publish_date == "2026-09-25T12:00:00Z"
    assert items[0].stance == StanceType.NEUTRAL

    # Check item 2 mapping
    assert items[1].id == "news_c1_2"
    assert items[1].publisher == "BBC News"
    assert items[1].domain == "bbc.com"


def test_missing_api_key_raises_error():
    """Tests that missing API key raises NewsAPIKeyError in live mode."""
    claim = ExtractedClaim(claim_id="c1", text="NASA")
    retriever = NewsAPIRetriever(api_key="", mock_mode=False)

    with pytest.raises(NewsAPIKeyError) as exc_info:
        retriever.search_claim_news(claim)

    assert "NEWS_API_KEY" in str(exc_info.value)


def test_http_401_unauthorized_error():
    """Tests handling of HTTP 401 invalid API key response."""
    claim = ExtractedClaim(claim_id="c1", text="NASA")
    retriever = NewsAPIRetriever(api_key="invalid_key", mock_mode=False)

    http_err = urllib.error.HTTPError(
        url="https://newsapi.org/v2/everything",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=MagicMock(read=MagicMock(return_value=json.dumps({
            "status": "error",
            "code": "apiKeyInvalid",
            "message": "Your API key is invalid or incorrect."
        }).encode("utf-8")))
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(NewsAPIError) as exc_info:
            retriever.search_claim_news(claim)

    assert exc_info.value.status_code == 401
    assert "invalid" in exc_info.value.message.lower()


def test_http_429_rate_limit_error():
    """Tests handling of HTTP 429 rate limit response."""
    claim = ExtractedClaim(claim_id="c1", text="NASA")
    retriever = NewsAPIRetriever(api_key="mock_key", mock_mode=False)

    http_err = urllib.error.HTTPError(
        url="https://newsapi.org/v2/everything",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=MagicMock(read=MagicMock(return_value=json.dumps({
            "status": "error",
            "code": "rateLimited",
            "message": "You have made too many requests recently."
        }).encode("utf-8")))
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(NewsAPIRateLimitError) as exc_info:
            retriever.search_claim_news(claim)

    assert exc_info.value.status_code == 429


def test_timeout_network_error():
    """Tests handling of gateway timeout raising NewsAPIError with code 504."""
    claim = ExtractedClaim(claim_id="c1", text="Federal Reserve")
    retriever = NewsAPIRetriever(api_key="mock_key", mock_mode=False)

    url_err = urllib.error.URLError(reason="timed out")

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(NewsAPIError) as exc_info:
            retriever.search_claim_news(claim)

    assert exc_info.value.status_code == 504
    assert "timeout" in exc_info.value.message.lower()


def test_malformed_json_response():
    """Tests handling of malformed non-JSON payload."""
    claim = ExtractedClaim(claim_id="c1", text="Apple")
    retriever = NewsAPIRetriever(api_key="mock_key", mock_mode=False)

    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(NewsAPIMalformedResponseError):
            retriever.search_claim_news(claim)


def test_empty_results_response():
    """Tests handling of empty articles array returning empty list."""
    claim = ExtractedClaim(claim_id="c1", text="Obscure Local Bakery Sourdough")
    retriever = NewsAPIRetriever(api_key="mock_key", mock_mode=False)

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"status": "ok", "totalResults": 0, "articles": []}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        items = retriever.search_claim_news(claim)

    assert isinstance(items, list)
    assert len(items) == 0


def test_removed_articles_filtered_out():
    """Tests that articles with title '[Removed]' are discarded."""
    claim = ExtractedClaim(claim_id="c1", text="Tech News")
    retriever = NewsAPIRetriever(api_key="mock_key", mock_mode=False)

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "status": "ok",
        "totalResults": 1,
        "articles": [
            {
                "source": {"name": "[Removed]"},
                "title": "[Removed]",
                "description": "[Removed]",
                "url": "https://removed.com",
                "publishedAt": "2026-09-25T00:00:00Z"
            }
        ]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        items = retriever.search_claim_news(claim)

    assert len(items) == 0


def test_verification_service_newsapi_integration():
    """
    Tests that VerificationService uses NewsAPIRetriever as the active live-news provider,
    safely handling NewsAPI failure while using Google Fact Check evidence.
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
            snippet="Reviewed Claim: 'James Webb Telescope discovered deepest infrared image' | Rating: True",
            stance=StanceType.SUPPORTS,
            raw_rating="True",
            claim_reviewed="James Webb Telescope discovered deepest infrared image"
        )
    ]

    mock_extractor = MagicMock()
    mock_extractor.extract_claims.return_value = [mock_claim]

    mock_fc_retriever = MagicMock()
    mock_fc_retriever.search_claim.return_value = mock_fc_evidence

    mock_news_retriever = MagicMock()
    mock_news_retriever.search_claim_news.side_effect = NewsAPIRateLimitError("Rate limit exceeded")

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
