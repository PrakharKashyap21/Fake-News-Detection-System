import re
from enum import Enum
from typing import List, Set, Optional, Dict, Any
from pydantic import BaseModel
from backend.app.v2.schemas import (
    ExtractedClaim,
    EvidenceItem,
    EvidenceSourceType,
    StanceType
)


class RelevanceClassification(str, Enum):
    RELEVANT = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"


class EvidenceMatchResult(BaseModel):
    relevance: RelevanceClassification
    stance: StanceType
    token_overlap_count: int = 0
    token_overlap_ratio: float = 0.0
    entity_overlap: List[str] = []
    number_overlap: List[str] = []
    date_overlap: List[str] = []
    negation_mismatch: bool = False
    matching_explanation: str = ""


class EvidenceMatcher:
    """Deterministic, inspectable claim-to-evidence relevance and stance matching engine."""

    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "when", "where", "how", "who", "which", "this", "that", "these", "those",
        "then", "just", "so", "than", "such", "both", "through", "about", "into",
        "over", "after", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "can",
        "could", "should", "may", "might", "shall", "must", "to", "from", "up",
        "down", "in", "out", "on", "off", "for", "with", "by", "at", "it", "its",
        "of", "on", "said", "says", "reported", "claims", "according"
    }

    NEGATION_WORDS = {
        "not", "no", "never", "cannot", "cant", "wont", "dont", "doesnt",
        "didnt", "isnt", "arent", "wasnt", "werent", "untrue", "false", "neither", "nor"
    }

    ACTION_VERBS = {
        "launched", "banned", "approved", "discovered", "cured", "declared",
        "signed", "taxed", "passed", "inaugurated", "found", "created", "built"
    }

    def _extract_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b[A-Za-z0-9'-]+\b", text)

    MONTHS_AND_DAYS = {
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    }

    def _extract_entities(self, text: str) -> List[str]:
        """Extracts proper nouns, acronyms, and capitalized terms excluding dates/months."""
        if not text:
            return []
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9'-]+\b|\b[A-Z]{2,}\b", text)
        seen = set()
        res = []
        for t in tokens:
            t_lower = t.lower()
            if t_lower not in self.STOPWORDS and t_lower not in self.MONTHS_AND_DAYS and t_lower not in seen and len(t) > 1:
                seen.add(t_lower)
                res.append(t)
        return res


    def _extract_numbers(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b\d+(?:\.\d+)?%?\b", text)

    def _extract_dates(self, text: str) -> List[str]:
        if not text:
            return []
        months = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
        dates = re.findall(rf"\b{months}\s+\d{{1,2}}\b|\b\d{{4}}\b", text, flags=re.IGNORECASE)
        return [d.lower() for d in dates]

    def _has_negation(self, text: str) -> bool:
        if not text:
            return False
        tokens = [t.lower() for t in self._extract_tokens(text)]
        for t in tokens:
            if t in self.NEGATION_WORDS or t.endswith("n't"):
                return True
        return False

    def match_evidence(self, claim: ExtractedClaim, item: EvidenceItem) -> EvidenceMatchResult:
        """Evaluates whether an evidence item is genuinely relevant to an extracted claim and determines stance."""
        if not claim or not claim.text or not item:
            return EvidenceMatchResult(
                relevance=RelevanceClassification.IRRELEVANT,
                stance=StanceType.NEUTRAL,
                matching_explanation="Empty claim or evidence item."
            )

        claim_text = claim.text
        evidence_text = f"{item.title or ''} {item.snippet or ''}".strip()

        # Tokens
        claim_tokens = [t.lower() for t in self._extract_tokens(claim_text)]
        evidence_tokens = [t.lower() for t in self._extract_tokens(evidence_text)]

        # Substantive non-date, non-stopword tokens
        substantive_claim = set(
            t for t in claim_tokens
            if t not in self.STOPWORDS and t not in self.MONTHS_AND_DAYS and not t.isdigit() and len(t) > 2
        )
        substantive_evidence = set(
            t for t in evidence_tokens
            if t not in self.STOPWORDS and t not in self.MONTHS_AND_DAYS and not t.isdigit() and len(t) > 2
        )


        token_overlap = substantive_claim.intersection(substantive_evidence)
        overlap_count = len(token_overlap)
        overlap_ratio = overlap_count / max(len(substantive_claim), 1)

        # Entities
        claim_entities = self._extract_entities(claim_text)
        evidence_entities_lower = set(e.lower() for e in self._extract_entities(evidence_text))
        matched_entities = [e for e in claim_entities if e.lower() in evidence_entities_lower]

        # Numbers
        claim_numbers = self._extract_numbers(claim_text)
        evidence_numbers_set = set(self._extract_numbers(evidence_text))
        matched_numbers = [n for n in claim_numbers if n in evidence_numbers_set]

        # Dates
        claim_dates = self._extract_dates(claim_text)
        evidence_dates_set = set(self._extract_dates(evidence_text))
        matched_dates = [d for d in claim_dates if d in evidence_dates_set]

        # Negations
        claim_negated = self._has_negation(claim_text)
        evidence_negated = self._has_negation(evidence_text)
        negation_mismatch = (claim_negated != evidence_negated)

        # Action verbs
        claim_actions = set(t for t in claim_tokens if t in self.ACTION_VERBS)
        evidence_actions = set(t for t in evidence_tokens if t in self.ACTION_VERBS)
        action_overlap = claim_actions.intersection(evidence_actions)

        # -------------------------------------------------------------
        # 1. DETERMINISTIC RELEVANCE CLASSIFICATION
        # -------------------------------------------------------------
        # Rule: Evidence is RELEVANT if:
        # - Has entity overlap (>= 1 proper noun entity matched) AND token overlap >= 1
        # - OR overlap ratio >= 0.35
        # - OR overlap count >= 3
        # - OR exact key action + entity match
        # Shared generic keyword alone or shared date alone does NOT make it relevant.

        is_relevant = False
        relevance_reason = ""

        if len(matched_entities) >= 1 and (overlap_count >= 1 or len(action_overlap) >= 1):
            is_relevant = True
            relevance_reason = f"Matched entities ({', '.join(matched_entities)}) with token overlap."
        elif overlap_count >= 3:
            is_relevant = True
            relevance_reason = f"High substantive token overlap ({overlap_count} terms)."
        elif len(action_overlap) >= 1 and len(matched_entities) >= 1:
            is_relevant = True
            relevance_reason = f"Matched action ({', '.join(action_overlap)}) and entity ({', '.join(matched_entities)})."
        elif overlap_ratio >= 0.4 and len(substantive_claim) >= 2:
            is_relevant = True
            relevance_reason = f"Substantive overlap ratio {overlap_ratio:.2f}."
        else:
            is_relevant = False
            relevance_reason = "Insufficient entity or action overlap; evidence is irrelevant to claim."

        if not is_relevant:
            return EvidenceMatchResult(
                relevance=RelevanceClassification.IRRELEVANT,
                stance=StanceType.NEUTRAL,
                token_overlap_count=overlap_count,
                token_overlap_ratio=round(overlap_ratio, 3),
                entity_overlap=matched_entities,
                number_overlap=matched_numbers,
                date_overlap=matched_dates,
                negation_mismatch=negation_mismatch,
                matching_explanation=relevance_reason
            )

        # -------------------------------------------------------------
        # 2. DETERMINISTIC STANCE DETERMINATION (SUPPORTS / CONTRADICTS / NEUTRAL)
        # -------------------------------------------------------------
        final_stance = StanceType.NEUTRAL
        stance_explanation = ""

        if item.source_type == EvidenceSourceType.FACT_CHECK_API:
            raw_stance = item.stance
            raw_rating_lower = str(item.raw_rating or "").lower()

            # Negation safety check for Fact Checks:
            # If claim is negated ("Government has NOT banned EV") and Fact Check rates "Government banned EV" as False:
            # Fact check refuted the positive statement -> so it SUPPORTS the negated claim!
            if claim_negated and raw_stance == StanceType.CONTRADICTS:
                final_stance = StanceType.SUPPORTS
                stance_explanation = "Fact check refuted the positive claim, supporting the negated claim."
            elif claim_negated and raw_stance == StanceType.SUPPORTS:
                final_stance = StanceType.CONTRADICTS
                stance_explanation = "Fact check verified positive claim, contradicting the negated claim."
            else:
                final_stance = raw_stance
                stance_explanation = f"Fact check provider rating ({item.raw_rating}) determined stance."

        elif item.source_type == EvidenceSourceType.LIVE_NEWS_SEARCH:
            # Live news default is NEUTRAL unless explicit entity + action + negation match exists
            if negation_mismatch:
                final_stance = StanceType.CONTRADICTS
                stance_explanation = "Negation mismatch between claim and live news article."
            elif len(action_overlap) >= 1 and len(matched_entities) >= 1 and not claim_negated:
                # Strong explicit action + entity match in live news coverage
                final_stance = StanceType.SUPPORTS
                stance_explanation = f"Live news explicitly reports action '{', '.join(action_overlap)}' for entity '{', '.join(matched_entities)}'."
            else:
                final_stance = StanceType.NEUTRAL
                stance_explanation = "Live news provides reporting context; stance remains neutral."

        return EvidenceMatchResult(
            relevance=RelevanceClassification.RELEVANT,
            stance=final_stance,
            token_overlap_count=overlap_count,
            token_overlap_ratio=round(overlap_ratio, 3),
            entity_overlap=matched_entities,
            number_overlap=matched_numbers,
            date_overlap=matched_dates,
            negation_mismatch=negation_mismatch,
            matching_explanation=f"{relevance_reason} Stance: {final_stance.value} ({stance_explanation})"
        )

    def process_claim_evidence(
        self, claim: ExtractedClaim, evidence_items: List[EvidenceItem]
    ) -> List[EvidenceItem]:
        """Filters out irrelevant evidence items and updates stances for relevant items."""
        if not claim or not evidence_items:
            return []

        processed: List[EvidenceItem] = []

        for item in evidence_items:
            match_res = self.match_evidence(claim, item)
            if match_res.relevance == RelevanceClassification.RELEVANT:
                # Update item stance based on matcher evaluation
                item_copy = item.model_copy(update={"stance": match_res.stance})
                processed.append(item_copy)

        return processed


_evidence_matcher_instance = None


def get_evidence_matcher() -> EvidenceMatcher:
    global _evidence_matcher_instance
    if _evidence_matcher_instance is None:
        _evidence_matcher_instance = EvidenceMatcher()
    return _evidence_matcher_instance
