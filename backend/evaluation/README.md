# Stage 26 — Real-World Evaluation Report

This directory contains the dataset, execution engine, results schema, and methodology for evaluating the Real-Time News Verification System (V2 Pipeline) against established ground truth and comparing it with the legacy V1 LinearSVM text classifier.

---

## 1. Methodology

The evaluation executes end-to-end API verification calls across the FastAPI application (`POST /v2/verify` and `POST /predict`) using a structured benchmark set of 24 real-world and synthetic evaluation cases.

### Evaluation Workflow:
1. **Case Submission**: Each test case (title + text) is submitted to the V2 verification endpoint (`POST /v2/verify`).
2. **Feature Extraction & Retrieval**:
   - Claims are extracted using the NLP `ClaimExtractor`.
   - Fact-checks are queried via `GoogleFactCheckRetriever`.
   - Live news coverage is queried via `GDELTNewsRetriever`.
3. **Aggregation & Verdict Synthesis**:
   - Evidence is aggregated by `EvidenceAggregator`.
   - Verdicts are evaluated deterministically by `VerdictEngine`.
   - Article-level overall assessment is synthesized conservatively.
4. **V1 SVM Comparison**: The identical case is submitted to `POST /predict` to capture V1 decision-margin prediction and confidence.
5. **Metric Calculation**: Results are compared against defensible ground truth labels and categorized.

---

## 2. Dataset Categories

The benchmark dataset (`dataset.json`) comprises **24 cases** across 8 distinct evaluation categories:

| Category | Description | Purpose |
|---|---|---|
| **1. KNOWN REAL CLAIMS** | Established true news events (e.g. NASA Mars Perseverance discovery). | Verify that legitimate news receives SUPPORTED or UNVERIFIED coverage without false contradictions. |
| **2. KNOWN FALSE CLAIMS** | Established false claims fact-checked by Snopes/PolitiFact (e.g. Pope Puffer Jacket). | Verify that debunked rumors receive CONTRADICTED verdicts. |
| **3. FACT-CHECKED CLAIMS** | Claims indexed in Google Fact Check API databases (e.g. 5G coronavirus myth). | Test direct Fact Check API retrieval hit rate and raw rating mapping. |
| **4. CURRENT NEWS CLAIMS** | Recent news reporting expected to match live GDELT search index. | Test live-news context retrieval without misinterpreting neutral coverage as proof of truth. |
| **5. REALISTIC SYNTHETIC FALSE CLAIMS** | Well-formulated false claims styled with formal journalistic phrasing (e.g. EU mandatory microchips). | Demonstrate that V2 relies on external evidence rather than text style, unlike V1 text classification. |
| **6. UNVERIFIED / LOW-EVIDENCE CLAIMS** | Obscure, local, or novel claims with zero external reporting. | Ensure system defaults to UNVERIFIED with HIGH uncertainty rather than hallucinating evidence. |
| **7. CONFLICTING-EVIDENCE CASES** | Controversial claims where opposing studies or reports coexist. | Test conflict detection flag (`has_conflict = True`) and conservative UNVERIFIED synthesis. |
| **8. MULTI-CLAIM ARTICLES** | Articles combining both supported and contradicted statements. | Verify multi-claim extraction and conservative overall assessment synthesis. |

---

## 3. Evaluation Metrics

- **Claim Extraction Success Rate**: Percentage of evaluation cases where at least one factual claim was successfully extracted.
- **Fact-Check Retrieval Hit Rate**: Percentage of cases returning relevant Fact-Check API evidence.
- **Live-News Retrieval Hit Rate**: Percentage of cases returning live GDELT news reporting.
- **Verdict Coverage Rate**: Percentage of cases yielding a structured verification response without system errors.
- **Ground-Truth Verdict Accuracy Rate**: Accuracy calculated strictly over cases with a defensible ground-truth label.
- **UNVERIFIED Assessment Rate**: Proportion of cases evaluated as UNVERIFIED due to evidence absence or conflict.
- **False-Positive Rate**: Percentage of false claims incorrectly evaluated as `SUPPORTED`.
- **False-Negative Rate**: Percentage of true claims incorrectly evaluated as `CONTRADICTED`.
- **Conflict Detection Count**: Frequency of `has_conflict` flag activations.
- **Average & Median Response Time**: Performance latency in milliseconds per verification request.

---

## 4. Limitations & Failure Categories

Observed evaluation failures are categorized into distinct operational buckets:

1. **`claim_extraction_failure`**: Sentence structure prevented rule-based claim extraction.
2. **`fact_check_miss`**: Debunked claim was not matched by Fact Check API search query parameters.
3. **`live_news_miss`**: Active news event was not indexed or retrieved by GDELT keywords.
4. **`source_conflict`**: Coexisting supporting and contradicting sources led to UNVERIFIED output.
5. **`insufficient_evidence`**: Expected result due to absence of indexed external reporting.
6. **`service_error`**: External API rate limits or HTTP errors (handled gracefully via fallback).

---

## 5. Important Interpretation Guidance

> [!IMPORTANT]
> **Statistical Disclaimer**: This 24-case evaluation dataset is a curated diagnostic benchmark designed to analyze functional pipeline behavior, retrieval hit rates, and V1 vs V2 differences. **It is not a statistically representative benchmark of all global web news.** High accuracy on this curated set demonstrates component integration integrity, but does not guarantee identical accuracy across arbitrary domain distributions.
