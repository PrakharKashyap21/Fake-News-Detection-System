import re
from typing import List, Set, Optional
from backend.app.v2.schemas import ExtractedClaim, EvidenceItem, StanceType

class FactCheckQueryBuilder:
    """Deterministic query builder and relevance filter for Google Fact Check Search API."""

    # Reporting Scaffolding to remove from primary queries
    SCAFFOLDING_PATTERNS = [
        r"\baccording to [^,.:]+,?\b",
        r"\bit is reported that\b",
        r"\bit has been reported that\b",
        r"\bofficials (said|say|stated|announced)\b",
        r"\bexperts (say|claim|stated|warn)\b",
        r"\bpeople (claim|believe|think)\b",
        r"\bstudies (show|suggest|indicate) that\b",
        r"\breports (allege|claim|state) that\b",
        r"\bbreaking news:?\b",
        r"\bsources (say|claim|report)\b",
        r"\bit was confirmed that\b",
        r"\bunverified assertion:?\b",
        r"\ba (photograph|photo|picture) (showing|of)\b",
        r"\bviral (social media )?posts (falsely )?(claimed|alleged|state) that\b",
        r"\bclaims that\b",
        r"\ballegedly\b",
        r"\breportedly\b",
        r"\bsupposedly\b",
    ]

    # Stopwords to filter out during query refinement (EXCLUDING negation words)
    LOW_INFO_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "when", "where", "how", "who", "which", "this", "that", "these", "those",
        "then", "just", "so", "than", "such", "both", "through", "about", "into",
        "over", "after", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "can",
        "could", "should", "may", "might", "shall", "must", "to", "from", "up",
        "down", "in", "out", "on", "off", "for", "with", "by", "at", "it", "its",
        "of", "said", "says", "reported", "claims", "also"
    }

    # Negation words MUST BE PRESERVED
    NEGATION_WORDS = {
        "not", "no", "never", "false", "cannot", "cant", "wont", "dont",
        "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "untrue", "neither", "nor"
    }

    def _clean_scaffolding(self, text: str) -> str:
        clean = text.strip()
        for pattern in self.SCAFFOLDING_PATTERNS:
            clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
        # Clean double spaces
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def _extract_negations(self, text: str) -> Set[str]:
        words = re.findall(r"\b[A-Za-z']+\b", text.lower())
        return {w for w in words if w in self.NEGATION_WORDS or w.endswith("n't")}

    def build_primary_query(self, claim: ExtractedClaim) -> str:
        """Constructs a deterministic, high-precision primary query string from the claim."""
        if not claim or not claim.text:
            return ""

        raw_text = claim.text.strip()
        # 1. Strip reporting scaffolding
        clean_text = self._clean_scaffolding(raw_text)
        if not clean_text:
            clean_text = raw_text

        # 2. Extract tokens preserving entities, action verbs, substantive objects, numbers, and negations
        tokens = re.findall(r"\b[A-Za-z0-9'-]+\b", clean_text)
        refined_tokens = []

        for token in tokens:
            t_lower = token.lower()
            # Preserve negations
            if t_lower in self.NEGATION_WORDS or t_lower.endswith("n't"):
                refined_tokens.append(token)
            # Preserve capitalized entities / acronyms / digits
            elif token[0].isupper() or any(char.isdigit() for char in token):
                refined_tokens.append(token)
            # Preserve substantive words
            elif len(t_lower) > 2 and t_lower not in self.LOW_INFO_WORDS:
                refined_tokens.append(token)

        # 3. Fallback to clean_text tokens if over-filtered
        if len(refined_tokens) < 2:
            refined_tokens = tokens

        # Take the top 8 most informative tokens
        query_tokens = refined_tokens[:8]
        query = " ".join(query_tokens)

        # 4. Enforce max query length (bounded at 90 chars for search precision)
        if len(query) > 90:
            query = query[:90].rsplit(" ", 1)[0]

        return query.strip()

    def build_fallback_query(self, claim: ExtractedClaim) -> str:
        """Constructs a deterministic secondary/fallback query from keywords and entities."""
        if not claim:
            return ""

        keywords = claim.keywords if claim.keywords else []
        text = claim.text or ""

        negations = self._extract_negations(text)
        
        fallback_tokens = []
        # Include explicit negations first if present in claim text
        for neg in sorted(list(negations)):
            fallback_tokens.append(neg)

        # Include extracted keywords
        for kw in keywords:
            if kw.lower() not in [t.lower() for t in fallback_tokens]:
                fallback_tokens.append(kw)

        # Include proper nouns / digits from original text if keywords are sparse
        if len(fallback_tokens) < 3 and text:
            entities = re.findall(r"\b[A-Z][a-z0-9]+\b|\b\d+\b", text)
            for ent in entities:
                if ent.lower() not in [t.lower() for t in fallback_tokens]:
                    fallback_tokens.append(ent)

        query = " ".join(fallback_tokens[:6])
        if len(query) > 80:
            query = query[:80].rsplit(" ", 1)[0]

        return query.strip()

    def filter_relevant_evidence(
        self, claim: ExtractedClaim, evidence_items: List[EvidenceItem]
    ) -> List[EvidenceItem]:
        """Applies deterministic relevance filtering on retrieved evidence items."""
        if not claim or not evidence_items:
            return []

        # Use EvidenceMatcher to perform exact claim and entity+predicate+object relevance filtering
        from backend.app.v2.evidence_matcher import get_evidence_matcher, RelevanceClassification
        matcher = get_evidence_matcher()

        filtered_items: List[EvidenceItem] = []
        for item in evidence_items:
            match_res = matcher.match_evidence(claim, item)
            if match_res.relevance == RelevanceClassification.RELEVANT:
                # Update item with matched stance and relevance score
                item_copy = item.model_copy(update={
                    "stance": match_res.stance,
                    "relevance_score": match_res.token_overlap_ratio
                })
                filtered_items.append(item_copy)

        return filtered_items


_query_builder_instance = None

def get_query_builder() -> FactCheckQueryBuilder:
    global _query_builder_instance
    if _query_builder_instance is None:
        _query_builder_instance = FactCheckQueryBuilder()
    return _query_builder_instance
