import re
from typing import List, Set, Optional, Dict, Tuple
from pydantic import BaseModel
from backend.app.v2.schemas import (
    ExtractedClaim,
    EvidenceItem,
    EvidenceSourceType,
    StanceType
)


class StanceAnalysisResult(BaseModel):
    stance: StanceType
    explanation: str
    modality_detected: Optional[str] = None
    negation_mismatch: bool = False
    action_conflict_detected: bool = False


class EvidenceStanceAnalyzer:
    """Deterministic, inspectable claim-to-evidence stance analysis engine.

    Evaluates proposition agreement, contradiction, modality, temporal language,
    negation alignment, and explicit action predicate conflicts.
    """

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

    MONTHS_AND_DAYS = {
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    }

    # Speculative / future / intention / discussion modality terms
    FUTURE_INTENT_MODALITY = {
        "plan", "plans", "planning", "planned", "will", "may", "might", "could",
        "expected", "expects", "expecting", "aims", "aiming", "proposes", "proposing",
        "considering", "considers", "discuss", "discusses", "discussing", "discussed",
        "schedule", "schedules", "scheduled", "scheduling", "upcoming", "future",
        "contemplating", "intends", "intent", "intention"
    }

    # Explicit negation and denial terms
    NEGATION_TERMS = {
        "not", "no", "never", "cannot", "can't", "cant", "wont", "won't",
        "dont", "don't", "doesnt", "doesn't", "didnt", "didn't", "isnt", "isn't",
        "arent", "aren't", "wasnt", "wasn't", "werent", "weren't", "hasnt", "hasn't",
        "false", "untrue", "denied", "denies", "deny", "denying", "refuted", "refutes",
        "debunked", "debunks", "neither", "nor", "myth"
    }

    # Explicit Action / Predicate Conflict Mappings
    # Format: verb -> set of opposing verbs/action concepts
    CONFLICTING_ACTIONS = {
        "launched": {"postponed", "cancelled", "canceled", "delayed", "aborted", "halted", "failed", "scrubbed"},
        "launch": {"postpone", "cancel", "delay", "abort", "halt", "scrub"},
        "approved": {"rejected", "denied", "disapproved", "blocked", "vetoed", "opposed", "cancelled", "canceled", "withdrawn", "scrapped"},
        "approve": {"reject", "deny", "disapprove", "block", "veto", "cancel"},
        "won": {"lost", "conceded", "defeated"},
        "win": {"lose", "concede"},
        "increased": {"decreased", "dropped", "fell", "reduced", "declined", "lowered"},
        "increase": {"decrease", "drop", "fall", "reduce", "decline", "lower"},
        "opened": {"closed", "shut", "sealed"},
        "open": {"close", "shut"},
        "acquired": {"denied", "rejected", "failed"},
        "acquire": {"deny", "reject"},
        "confirmed": {"denied", "refuted", "debunked", "disproved"},
        "confirm": {"deny", "refute", "debunk", "disprove"},
        "passed": {"failed", "rejected", "blocked", "defeated"},
        "pass": {"fail", "reject", "block"},
        "banned": {"allowed", "permitted", "unbanned", "lifted", "approved"},
        "ban": {"allow", "permit", "unban", "lift"}
    }

    ACTION_VERBS = {
        "launched", "launch", "approved", "approve", "won", "win",
        "increased", "increase", "opened", "open", "acquired", "acquire",
        "confirmed", "confirm", "passed", "pass", "banned", "ban",
        "discovered", "discover", "built", "build", "signed", "sign"
    }

    def _extract_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b[A-Za-z0-9'-]+\b", text)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract proper nouns, acronyms, and capitalized terms excluding months/weekdays."""
        if not text:
            return []
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9'-]+\b|\b[A-Z]{2,}\b", text)
        seen = set()
        res = []
        for t in tokens:
            t_lower = t.lower()
            if (
                t_lower not in self.STOPWORDS
                and t_lower not in self.MONTHS_AND_DAYS
                and t_lower not in seen
                and len(t) > 1
            ):
                seen.add(t_lower)
                res.append(t)
        return res

    def _has_negation(self, text: str) -> bool:
        if not text:
            return False
        tokens = [t.lower() for t in self._extract_tokens(text)]
        for t in tokens:
            if t in self.NEGATION_TERMS or t.endswith("n't"):
                return True
        return False

    def _detect_future_or_speculative_modality(self, text: str) -> Optional[str]:
        """Detects if text uses future, speculative, or intention modality."""
        if not text:
            return None
        tokens = [t.lower() for t in self._extract_tokens(text)]
        for i, t in enumerate(tokens):
            if t in self.FUTURE_INTENT_MODALITY:
                # E.g., "plans to launch", "will launch", "discusses plans"
                context = t
                if i + 1 < len(tokens):
                    context = f"{t} {tokens[i+1]}"
                return context
        return None

    def _check_action_conflict(self, claim_text: str, evidence_text: str) -> Tuple[bool, str]:
        """Checks if claim action conflicts explicitly with evidence action."""
        claim_tokens = [t.lower() for t in self._extract_tokens(claim_text)]
        evidence_tokens = [t.lower() for t in self._extract_tokens(evidence_text)]
        evidence_tokens_set = set(evidence_tokens)

        for c_token in claim_tokens:
            if c_token in self.CONFLICTING_ACTIONS:
                opposing_set = self.CONFLICTING_ACTIONS[c_token]
                matched_conflict = opposing_set.intersection(evidence_tokens_set)
                if matched_conflict:
                    opposing_verb = list(matched_conflict)[0]
                    return True, f"Action conflict: claim '{c_token}' vs evidence '{opposing_verb}'"

            # Check reverse conflict (claim has postponed, evidence has launched)
            for verb, opposing_set in self.CONFLICTING_ACTIONS.items():
                if c_token in opposing_set and verb in evidence_tokens_set:
                    return True, f"Action conflict: claim '{c_token}' vs evidence '{verb}'"

        return False, ""

    def analyze_stance(self, claim: ExtractedClaim, item: EvidenceItem) -> StanceAnalysisResult:
        """Analyzes proposition structure to assign SUPPORTS, CONTRADICTS, or NEUTRAL stance."""
        if not claim or not claim.text or not item:
            return StanceAnalysisResult(
                stance=StanceType.NEUTRAL,
                explanation="Empty claim or evidence item."
            )

        # 1. Fact Check API evidence preservation
        if item.source_type == EvidenceSourceType.FACT_CHECK_API:
            claim_negated = self._has_negation(claim.text)
            raw_stance = item.stance
            if claim_negated and raw_stance == StanceType.CONTRADICTS:
                return StanceAnalysisResult(
                    stance=StanceType.SUPPORTS,
                    explanation="Fact check refuted positive statement, supporting negated claim.",
                    negation_mismatch=False
                )
            elif claim_negated and raw_stance == StanceType.SUPPORTS:
                return StanceAnalysisResult(
                    stance=StanceType.CONTRADICTS,
                    explanation="Fact check verified positive statement, contradicting negated claim.",
                    negation_mismatch=True
                )
            return StanceAnalysisResult(
                stance=raw_stance,
                explanation=f"Fact check provider rating ({item.raw_rating}) preserved."
            )

        # 2. Live News / General Evidence Stance Analysis
        claim_text = claim.text
        evidence_text = f"{item.title or ''} {item.snippet or ''}".strip()

        claim_negated = self._has_negation(claim_text)
        evidence_negated = self._has_negation(evidence_text)

        claim_entities = self._extract_entities(claim_text)
        evidence_entities = set(e.lower() for e in self._extract_entities(evidence_text))
        matched_entities = [e for e in claim_entities if e.lower() in evidence_entities]

        claim_modality = self._detect_future_or_speculative_modality(claim_text)
        evidence_modality = self._detect_future_or_speculative_modality(evidence_text)

        has_action_conflict, conflict_reason = self._check_action_conflict(claim_text, evidence_text)

        # Modality check:
        # If claim asserts a completed action (no future modality) but evidence uses future/intent modality
        # (e.g. claim="Company launched Product X", evidence="Company plans to launch Product X"):
        # Evidence stance MUST BE NEUTRAL (future intention is not completed event confirmation).
        if not claim_modality and evidence_modality:
            return StanceAnalysisResult(
                stance=StanceType.NEUTRAL,
                explanation=f"Claim asserts completed event, but evidence uses future/intent modality '{evidence_modality}'. Stance: NEUTRAL.",
                modality_detected=evidence_modality
            )

        # Action Conflict (e.g. launched vs postponed/cancelled, approved vs rejected, won vs lost)
        if has_action_conflict:
            return StanceAnalysisResult(
                stance=StanceType.CONTRADICTS,
                explanation=f"{conflict_reason}. Stance: CONTRADICTS.",
                action_conflict_detected=True
            )

        # Explicit Negation Alignment / Disagreement
        # Case A: Claim is positive ("Apple acquired Company X") and evidence is negated ("Apple did not acquire Company X" / "Apple denied acquiring Company X")
        if not claim_negated and evidence_negated:
            # Verify entity match exists to prevent false contradiction from generic 'false' elsewhere
            if len(matched_entities) >= 1 or len(claim_entities) == 0:
                return StanceAnalysisResult(
                    stance=StanceType.CONTRADICTS,
                    explanation="Evidence explicitly negates or denies the positive claim proposition. Stance: CONTRADICTS.",
                    negation_mismatch=True
                )
            else:
                return StanceAnalysisResult(
                    stance=StanceType.NEUTRAL,
                    explanation="Negation present in evidence but entity alignment is insufficient. Stance: NEUTRAL."
                )

        # Case B: Claim is negated ("Apple did not acquire Company X") and evidence is also negated/denying ("Apple denied acquiring Company X")
        if claim_negated and evidence_negated:
            if len(matched_entities) >= 1 or len(claim_entities) == 0:
                return StanceAnalysisResult(
                    stance=StanceType.SUPPORTS,
                    explanation="Evidence agrees with the negated claim proposition (denial/negation match). Stance: SUPPORTS.",
                    negation_mismatch=False
                )

        # Case C: Claim is negated ("Apple did not acquire Company X") and evidence asserts positive action ("Apple acquired Company X")
        if claim_negated and not evidence_negated:
            if len(matched_entities) >= 1 or len(claim_entities) == 0:
                return StanceAnalysisResult(
                    stance=StanceType.CONTRADICTS,
                    explanation="Evidence asserts positive event for negated claim proposition. Stance: CONTRADICTS.",
                    negation_mismatch=True
                )

        # Proposition Agreement (Direct Support)
        claim_tokens = [t.lower() for t in self._extract_tokens(claim_text)]
        evidence_tokens = set(t.lower() for t in self._extract_tokens(evidence_text))

        claim_actions = set(t for t in claim_tokens if t in self.ACTION_VERBS)
        evidence_actions = set(t for t in evidence_tokens if t in self.ACTION_VERBS)

        # If claim contains explicit action verbs, evidence MUST share at least one action verb to SUPPORT
        if claim_actions and not claim_actions.intersection(evidence_actions):
            return StanceAnalysisResult(
                stance=StanceType.NEUTRAL,
                explanation=f"Claim action predicate '{', '.join(claim_actions)}' is missing from evidence. Stance: NEUTRAL."
            )

        # Check key action/meaningful terms overlap
        entity_token_lowers = set(e.lower() for e in claim_entities)
        meaningful_claim_terms = set(
            t for t in claim_tokens
            if t not in self.STOPWORDS and t not in self.MONTHS_AND_DAYS and t not in entity_token_lowers and not t.isdigit() and len(t) > 2
        )
        overlap = meaningful_claim_terms.intersection(evidence_tokens)

        # Direct support criteria:
        # 1. Matched entities >= 1 (or no entities in claim)
        # 2. Substantive non-entity overlap >= 1 term
        if (len(matched_entities) >= 1 or len(claim_entities) == 0) and (len(overlap) >= 1 or len(claim_actions.intersection(evidence_actions)) >= 1):
            return StanceAnalysisResult(
                stance=StanceType.SUPPORTS,
                explanation=f"Direct proposition support: matched entities ({', '.join(matched_entities)}) and key terms ({', '.join(overlap)}). Stance: SUPPORTS."
            )

        # Default Fallback for topically related / insufficient evidence
        return StanceAnalysisResult(
            stance=StanceType.NEUTRAL,
            explanation="Evidence is topically related but does not provide sufficient explicit proposition confirmation. Stance: NEUTRAL."
        )

    def process_claim_evidence_stance(
        self, claim: ExtractedClaim, evidence_items: List[EvidenceItem]
    ) -> List[EvidenceItem]:
        """Analyzes stance for relevant evidence items and returns updated EvidenceItems."""
        if not claim or not evidence_items:
            return []

        analyzed: List[EvidenceItem] = []
        for item in evidence_items:
            result = self.analyze_stance(claim, item)
            item_copy = item.model_copy(update={"stance": result.stance})
            analyzed.append(item_copy)

        return analyzed


_stance_analyzer_instance = None


def get_stance_analyzer() -> EvidenceStanceAnalyzer:
    global _stance_analyzer_instance
    if _stance_analyzer_instance is None:
        _stance_analyzer_instance = EvidenceStanceAnalyzer()
    return _stance_analyzer_instance
