import pytest
from backend.app.v2.schemas import (
    ExtractedClaim,
    EvidenceItem,
    EvidenceSourceType,
    StanceType,
    ClaimVerdict,
    VerificationRequest
)
from backend.app.v2.evidence_matcher import EvidenceMatcher, RelevanceClassification
from backend.app.v2.query_builder import FactCheckQueryBuilder
from backend.app.v2.verification_service import VerificationService


@pytest.fixture
def matcher():
    return EvidenceMatcher()


@pytest.fixture
def query_builder():
    return FactCheckQueryBuilder()


def test_exact_same_claim_fact_check_accepted(matcher):
    """Test 1: Exact / same-claim fact-check is accepted as RELEVANT with correct stance."""
    claim = ExtractedClaim(
        claim_id="c1",
        text="UNESCO declared Jana Gana Mana as the best national anthem in the world.",
        keywords=["UNESCO", "Jana Gana Mana", "national anthem"]
    )
    evidence = EvidenceItem(
        id="fc_1",
        claim_id="c1",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="Factly",
        domain="factly.in",
        url="https://factly.in/unesco-anthem",
        title="Fact Check: UNESCO national anthem announcement",
        snippet="Reviewed Claim: 'Did UNESCO declare India's Jana Gana Mana the best national anthem in the world?' | Rating: False",
        stance=StanceType.CONTRADICTS,
        raw_rating="False",
        claim_reviewed="Did UNESCO declare India's Jana Gana Mana as the best national anthem in the world?"
    )

    result = matcher.match_evidence(claim, evidence)
    assert result.relevance == RelevanceClassification.RELEVANT
    assert result.stance == StanceType.CONTRADICTS
    assert "UNESCO" in result.entity_overlap


def test_clearly_unrelated_claim_rejected(matcher):
    """Test 2: Completely unrelated claim is rejected as IRRELEVANT."""
    claim = ExtractedClaim(
        claim_id="c2",
        text="NASA launched the James Webb Space Telescope into orbit.",
        keywords=["NASA", "James Webb", "Telescope"]
    )
    evidence = EvidenceItem(
        id="fc_2",
        claim_id="c2",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="Snopes",
        domain="snopes.com",
        url="https://snopes.com/gas-stoves",
        title="Fact Check: European Union gas stove ban",
        snippet="Reviewed Claim: 'Did the European Union ban all gas stoves in restaurants?' | Rating: False",
        stance=StanceType.CONTRADICTS,
        raw_rating="False",
        claim_reviewed="Did the European Union ban all gas stoves in restaurants?"
    )

    result = matcher.match_evidence(claim, evidence)
    assert result.relevance == RelevanceClassification.IRRELEVANT
    assert result.stance == StanceType.NEUTRAL
    assert "Rejected" in result.matching_explanation


def test_same_entities_different_predicate_rejected(matcher):
    """Test 3: Same entities (NASA, Mars) but completely different predicate/action (discovered bacteria vs photographed face) is rejected."""
    claim = ExtractedClaim(
        claim_id="c3",
        text="NASA discovered living bacteria on Mars.",
        keywords=["NASA", "Mars", "bacteria"]
    )
    evidence = EvidenceItem(
        id="fc_3",
        claim_id="c3",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="USA Today",
        domain="usatoday.com",
        url="https://usatoday.com/mars-face",
        title="Fact Check: NASA photographed pyramid on Mars",
        snippet="Reviewed Claim: 'NASA photographed a strange alien face and pyramid structure on Mars' | Rating: False",
        stance=StanceType.CONTRADICTS,
        raw_rating="False",
        claim_reviewed="NASA photographed a strange alien face and pyramid structure on Mars"
    )

    result = matcher.match_evidence(claim, evidence)
    assert result.relevance == RelevanceClassification.IRRELEVANT
    assert result.stance == StanceType.NEUTRAL
    assert "Rejected adjacent fact-check" in result.matching_explanation


def test_same_topic_materially_different_claim_rejected(matcher):
    """Test 4: Same broad topic (CERN/LHC) but materially different claim (Higgs boson discovery vs creating black hole) is rejected."""
    claim = ExtractedClaim(
        claim_id="c4",
        text="The Large Hadron Collider discovered the Higgs boson in 2012.",
        keywords=["Large Hadron Collider", "Higgs boson", "2012"]
    )
    evidence = EvidenceItem(
        id="fc_4",
        claim_id="c4",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="AFP Fact Check",
        domain="factcheck.afp.com",
        url="https://factcheck.afp.com/cern-portal",
        title="Fact Check: CERN portal black hole",
        snippet="Reviewed Claim: 'CERN scientists created a black hole that will destroy Earth' | Rating: False",
        stance=StanceType.CONTRADICTS,
        raw_rating="False",
        claim_reviewed="CERN scientists created a black hole that will destroy Earth"
    )

    result = matcher.match_evidence(claim, evidence)
    assert result.relevance == RelevanceClassification.IRRELEVANT
    assert result.stance == StanceType.NEUTRAL
    assert "Rejected adjacent fact-check" in result.matching_explanation


def test_relevant_contradiction_accepted(matcher):
    """Test 5: Relevant contradiction of a debunked false claim is accepted with CONTRADICTS stance."""
    claim = ExtractedClaim(
        claim_id="c5",
        text="Drinking hot lemon water with baking soda completely cures cancer.",
        keywords=["lemon water", "baking soda", "cures cancer"]
    )
    evidence = EvidenceItem(
        id="fc_5",
        claim_id="c5",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="PolitiFact",
        domain="politifact.com",
        url="https://politifact.com/lemon-cancer",
        title="Fact Check: Lemon water cancer cure",
        snippet="Reviewed Claim: 'Drinking hot lemon water with baking soda cures cancer 10,000 times better than chemo' | Rating: False",
        stance=StanceType.CONTRADICTS,
        raw_rating="False",
        claim_reviewed="Drinking hot lemon water with baking soda cures cancer 10000 times better than chemotherapy"
    )

    result = matcher.match_evidence(claim, evidence)
    assert result.relevance == RelevanceClassification.RELEVANT
    assert result.stance == StanceType.CONTRADICTS


def test_relevant_support_accepted(matcher):
    """Test 6: Relevant fact check verifying true claim is accepted with SUPPORTS stance."""
    claim = ExtractedClaim(
        claim_id="c6",
        text="WHO ended the COVID-19 global health emergency in May 2023.",
        keywords=["WHO", "COVID-19", "global health emergency", "May 2023"]
    )
    evidence = EvidenceItem(
        id="fc_6",
        claim_id="c6",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="Reuters Fact Check",
        domain="reuters.com",
        url="https://reuters.com/fact-check/who-covid-emergency",
        title="Fact Check: WHO COVID-19 emergency status",
        snippet="Reviewed Claim: 'The World Health Organization declared an end to the COVID-19 global health emergency in May 2023' | Rating: True",
        stance=StanceType.SUPPORTS,
        raw_rating="True",
        claim_reviewed="The World Health Organization declared an end to the COVID-19 global health emergency in May 2023"
    )

    result = matcher.match_evidence(claim, evidence)
    assert result.relevance == RelevanceClassification.RELEVANT
    assert result.stance == StanceType.SUPPORTS


def test_missing_claim_reviewed_handled_safely(matcher):
    """Test 7: When claim_reviewed is missing/None, falls back to title safely."""
    # Case 7a: Relevant title match
    claim = ExtractedClaim(
        claim_id="c7a",
        text="NASA's Apollo 11 mission landed humans on the Moon in 1969.",
        keywords=["NASA", "Apollo 11", "Moon", "1969"]
    )
    evidence_valid = EvidenceItem(
        id="fc_7a",
        claim_id="c7a",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="FactCheck.org",
        domain="factcheck.org",
        url="https://factcheck.org/apollo-11",
        title="Fact Check: Did the Apollo 11 mission land humans on the Moon in 1969?",
        snippet="Rating: True",
        stance=StanceType.SUPPORTS,
        raw_rating="True",
        claim_reviewed=None  # Missing
    )

    result_valid = matcher.match_evidence(claim, evidence_valid)
    assert result_valid.relevance == RelevanceClassification.RELEVANT
    assert result_valid.stance == StanceType.SUPPORTS

    # Case 7b: Irrelevant title match
    evidence_irrelevant = EvidenceItem(
        id="fc_7b",
        claim_id="c7a",
        source_type=EvidenceSourceType.FACT_CHECK_API,
        publisher="Snopes",
        domain="snopes.com",
        url="https://snopes.com/moon-cheese",
        title="Fact Check: Is the Moon made of green cheese?",
        snippet="Rating: False",
        stance=StanceType.CONTRADICTS,
        raw_rating="False",
        claim_reviewed=None  # Missing
    )

    result_irrel = matcher.match_evidence(claim, evidence_irrelevant)
    assert result_irrel.relevance == RelevanceClassification.IRRELEVANT
    assert result_irrel.stance == StanceType.NEUTRAL


def test_query_builder_focuses_on_extracted_claim(query_builder):
    """Test 8: Query builder generates concise queries preserving entities and key predicates."""
    claim = ExtractedClaim(
        claim_id="c8",
        text="According to viral social media reports, NASA discovered organic carbon molecules in Jezero Crater on Mars.",
        keywords=["NASA", "Perseverance", "Jezero Crater", "Mars"]
    )
    primary_query = query_builder.build_primary_query(claim)

    # Scaffolding must be stripped
    assert "according to" not in primary_query.lower()
    assert "viral social media" not in primary_query.lower()
    # High-value entities and objects must be present
    assert "NASA" in primary_query
    assert "Mars" in primary_query
    assert "discovered" in primary_query
    assert len(primary_query) <= 90
