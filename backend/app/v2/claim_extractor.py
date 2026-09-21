import re
from typing import List
from backend.app.v2.schemas import ExtractedClaim, ClaimVerdict


class ClaimExtractor:
    """Extracts atomic, verifiable factual claims and query keywords from article headlines and body text."""

    OPINION_INDICATORS = [
        r"\bi think\b", r"\bin my opinion\b", r"\bwe believe\b", r"\bi believe\b",
        r"\bit seems\b", r"\bi feel\b", r"\barguably\b", r"\bin my view\b",
        r"\bmade a mistake\b", r"\bpoliticians should\b", r"\bshould be\b"
    ]

    ALERT_HEADERS = {
        "breaking news", "breaking", "latest news", "just in", "news alert",
        "opinion", "editorial", "update", "shocking proof"
    }

    INTERJECTIONS = {
        "wow", "hello", "hi", "hey", "greetings", "congratulations", "oh", "gosh"
    }

    EXCLAMATORY_PATTERNS = [
        r"^what a\b", r"^how (wonderful|great|amazing|terrible|awesome)\b"
    ]

    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "when", "where", "how", "who", "which", "this", "that", "these", "those",
        "then", "just", "so", "than", "such", "both", "through", "about", "into",
        "over", "after", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "can",
        "could", "should", "may", "might", "shall", "must", "to", "from", "up",
        "down", "in", "out", "on", "off", "for", "with", "by", "at", "it", "its"
    }

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    def _is_opinion_or_non_factual(self, sentence: str) -> bool:
        s_clean = sentence.strip()
        s_lower = s_clean.lower()
        words = re.findall(r"\b[A-Za-z0-9\'-]+\b", s_lower)

        # 1. Less than 2 words is a non-factual fragment
        if len(words) < 2:
            return True

        # 2. Questions are not factual assertions
        if s_clean.endswith("?"):
            return True

        # 3. Known opinion / subjective indicators
        for pattern in self.OPINION_INDICATORS:
            if re.search(pattern, s_lower):
                return True

        # 4. Interjections & Greetings at start
        if words and words[0] in self.INTERJECTIONS:
            return True

        # 5. Generic alert headers (e.g., "Breaking news!", "Opinion:")
        clean_header = re.sub(r"[!:]$", "", s_lower).strip()
        if clean_header in self.ALERT_HEADERS or s_lower.startswith(("opinion:", "editorial:", "analysis:")):
            return True

        # 6. Exclamatory evaluation structures (e.g. "What a wonderful day...")
        for pattern in self.EXCLAMATORY_PATTERNS:
            if re.search(pattern, s_lower):
                return True

        # 7. Exclamatory / emotional reaction sentences without action verbs
        if s_clean.endswith("!") and any(w in s_lower for w in ["unbelievable", "wonderful", "terrible", "amazing", "great", "disaster", "shocking"]):
            return True

        return False

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r"\b[A-Za-z0-9\'-]+\b", text)
        keywords = []
        for w in words:
            w_clean = w.strip("'")
            if len(w_clean) > 2 and w_clean.lower() not in self.STOPWORDS:
                keywords.append(w_clean)
        # Deduplicate preserving order
        seen = set()
        unique_kw = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_kw.append(kw)
        return unique_kw[:8]

    def extract_claims(self, title: str = "", text: str = "", max_claims: int = 5) -> List[ExtractedClaim]:
        clean_title = self._clean_text(title)
        clean_body = self._clean_text(text)

        raw_sentences = []
        if clean_title:
            raw_sentences.append(clean_title)

        if clean_body:
            # Split body into sentences using basic punctuation regex
            body_sentences = re.split(r"(?<=[.!?])\s+", clean_body)
            raw_sentences.extend(body_sentences)

        claims: List[ExtractedClaim] = []
        seen_texts = set()

        for idx, sentence in enumerate(raw_sentences):
            sent = self._clean_text(sentence)
            if not sent:
                continue

            if self._is_opinion_or_non_factual(sent):
                continue

            sent_key = sent.lower()
            if sent_key in seen_texts:
                continue
            seen_texts.add(sent_key)

            keywords = self._extract_keywords(sent)
            claim_id = f"c{len(claims) + 1}"

            claims.append(
                ExtractedClaim(
                    claim_id=claim_id,
                    text=sent,
                    verdict=ClaimVerdict.UNVERIFIED,
                    support_score=0.0,
                    contradict_score=0.0,
                    keywords=keywords,
                    evidence=[],
                    explanation="Claim extracted for verification; evidence retrieval pending."
                )
            )

            if len(claims) >= max_claims:
                break

        return claims


_claim_extractor_instance = None


def get_claim_extractor() -> ClaimExtractor:
    global _claim_extractor_instance
    if _claim_extractor_instance is None:
        _claim_extractor_instance = ClaimExtractor()
    return _claim_extractor_instance
