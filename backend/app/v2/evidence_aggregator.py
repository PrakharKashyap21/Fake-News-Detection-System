import re
from typing import List, Optional, Set
from backend.app.v2.schemas import (
    EvidenceItem,
    ExtractedClaim,
    EvidenceSourceType,
    StanceType,
    ClaimEvidenceSummary
)


class EvidenceAggregator:
    """Aggregates and summarizes evidence from fact-check and live-news sources for an extracted claim."""

    def _normalize_url(self, url: Optional[str]) -> str:
        """Normalizes URL string for duplicate detection."""
        if not url:
            return ""
        clean = str(url).strip().lower()
        if clean.endswith("/"):
            clean = clean[:-1]
        return clean

    def _parse_timestamp(self, ts: Optional[str]) -> Optional[str]:
        """Validates timestamp string for deterministic recency comparison."""
        if not ts:
            return None
        clean = str(ts).strip()
        if len(clean) >= 4 and (clean[0:4].isdigit() or "T" in clean):
            return clean
        return None

    def aggregate_evidence(
        self,
        claim: ExtractedClaim,
        evidence_items: Optional[List[EvidenceItem]] = None
    ) -> ClaimEvidenceSummary:
        """Aggregates evidence items for a given claim and produces a structured summary."""
        if evidence_items is None:
            evidence_items = claim.evidence if claim and claim.evidence else []

        claim_id = claim.claim_id if claim and claim.claim_id else "unknown_claim"
        claim_text = claim.text if claim and claim.text else ""

        if not evidence_items:
            return ClaimEvidenceSummary(
                claim_id=claim_id,
                claim_text=claim_text,
                total_evidence_count=0,
                fact_check_count=0,
                live_news_count=0,
                supporting_evidence_count=0,
                contradicting_evidence_count=0,
                neutral_evidence_count=0,
                unique_source_domains=[],
                unique_domain_count=0,
                most_recent_evidence_timestamp=None,
                has_conflicting_evidence=False,
                fact_check_evidence=[],
                live_news_evidence=[],
                all_evidence=[]
            )

        # 1. Deduplicate by EvidenceItem ID and normalized URL
        seen_ids: Set[str] = set()
        seen_urls: Set[str] = set()
        deduped_evidence: List[EvidenceItem] = []

        for item in evidence_items:
            if not isinstance(item, EvidenceItem):
                continue
            item_id = str(item.id).strip() if item.id else ""
            if item_id and item_id in seen_ids:
                continue

            norm_url = self._normalize_url(item.url)
            if norm_url and norm_url in seen_urls:
                continue

            if item_id:
                seen_ids.add(item_id)
            if norm_url:
                seen_urls.add(norm_url)

            deduped_evidence.append(item)

        # 2. Separate by source type & stances
        fact_check_evidence: List[EvidenceItem] = []
        live_news_evidence: List[EvidenceItem] = []
        domains_set: Set[str] = set()

        supporting_count = 0
        contradicting_count = 0
        neutral_count = 0
        latest_timestamp: Optional[str] = None

        for item in deduped_evidence:
            if item.source_type == EvidenceSourceType.FACT_CHECK_API:
                fact_check_evidence.append(item)
            elif item.source_type == EvidenceSourceType.LIVE_NEWS_SEARCH:
                live_news_evidence.append(item)
            else:
                live_news_evidence.append(item)

            if item.domain:
                clean_domain = str(item.domain).strip().lower()
                if clean_domain and clean_domain != "unknown":
                    domains_set.add(clean_domain)

            if item.stance == StanceType.SUPPORTS:
                supporting_count += 1
            elif item.stance == StanceType.CONTRADICTS:
                contradicting_count += 1
            else:
                neutral_count += 1

            # Determine most recent publish timestamp
            ts = self._parse_timestamp(item.publish_date)
            if ts:
                if latest_timestamp is None or ts > latest_timestamp:
                    latest_timestamp = ts

        sorted_domains = sorted(list(domains_set))
        has_conflict = (supporting_count > 0 and contradicting_count > 0)

        return ClaimEvidenceSummary(
            claim_id=claim_id,
            claim_text=claim_text,
            total_evidence_count=len(deduped_evidence),
            fact_check_count=len(fact_check_evidence),
            live_news_count=len(live_news_evidence),
            supporting_evidence_count=supporting_count,
            contradicting_evidence_count=contradicting_count,
            neutral_evidence_count=neutral_count,
            unique_source_domains=sorted_domains,
            unique_domain_count=len(sorted_domains),
            most_recent_evidence_timestamp=latest_timestamp,
            has_conflicting_evidence=has_conflict,
            fact_check_evidence=fact_check_evidence,
            live_news_evidence=live_news_evidence,
            all_evidence=deduped_evidence
        )


_evidence_aggregator_instance = None


def get_evidence_aggregator() -> EvidenceAggregator:
    global _evidence_aggregator_instance
    if _evidence_aggregator_instance is None:
        _evidence_aggregator_instance = EvidenceAggregator()
    return _evidence_aggregator_instance
