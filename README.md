# Real-Time News & Claim Verification System

A multi-stage news and claim verification system combining real-time external evidence retrieval (Google Fact Check Tools API, Live News) with relevance filtering, stance analysis, automated verdict synthesis, and auxiliary linguistic style classification.

---

## 1. Project Overview

The **Real-Time News & Claim Verification System** is designed to evaluate the factual accuracy of news headlines and article claims against authoritative external evidence.

While traditional machine learning approaches detect misinformation solely by identifying lexical or stylistic writing patterns, this system implements an evidence-first architecture: extracting key factual assertions, querying fact-checking databases and live news sources, evaluating evidence relevance and stance, and producing a grounded verdict (`SUPPORTED`, `CONTRADICTED`, or `UNVERIFIED`).

---

## 2. Evolution: V1 Linguistic Classifier to V2 Evidence Pipeline

### V1 Architecture (Stylistic Text Classification)
The initial V1 implementation was a standalone machine learning text classifier:
- **Pipeline**: Text Preprocessing → TF-IDF Unigram/Bigram Vectorization → LinearSVC Classifier trained on the WELFake dataset (63,672 cleaned records).
- **Function**: Detected statistical writing style patterns associated with fake vs. real news.

### Why V2 Was Introduced
Linguistic style is not a proxy for factual truth:
- Fabricated claims or deliberate misinformation can be drafted in professional, formal journalistic prose.
- Legitimate, breaking, or informal reporting may exhibit stylistic traits that a static text classifier flags as suspicious.
- Static classifiers cannot verify dynamic real-world facts, ongoing events, or check claims against fact-checking publishers.

To address these fundamental limitations, **V2** introduced a multi-source evidence verification pipeline. The V1 LinearSVC model was repurposed **strictly as an auxiliary linguistic signal** that provides stylistic context without determining truth.

---

## 3. Current V2 Verification Pipeline

The V2 pipeline evaluates news claims across six sequential stages:

```
User Input (Headline & Article Text)
  │
  ▼
1. Claim Extraction ──────────► Extracts core factual assertions & generates search queries
  │
  ▼
2. Evidence Retrieval ────────► Queries Google Fact Check Tools API & Live News (GDELT)
  │
  ▼
3. Relevance Matching ────────► Filters retrieved evidence by text similarity & keyword overlap
  │
  ▼
4. Stance Analysis ───────────► Evaluates whether evidence SUPPORTS, CONTRADICTS, or is NEUTRAL
  │
  ▼
5. Auxiliary Signals ─────────► Incorporates V1 LinearSVC linguistic signal as secondary context
  │
  ▼
6. Verdict Engine ────────────► Synthesizes evidence & stance balance into final verdict
  │
  ▼
Output: Final Verdict (SUPPORTED / CONTRADICTED / UNVERIFIED) + Evidence Details + Confidence
```

### Pipeline Breakdown:
1. **Claim Extraction**: Parses submitted text into discrete, verifiable factual claims and formulates search queries.
2. **Fact-Check Retrieval**: Concurrently queries external providers:
   - **Google Fact Check Tools API**: Retrieves published assessments from fact-checking publishers (Snopes, PolitiFact, Full Fact, BOOM, AFP, etc.).
   - **Live News Retrieval (GDELT)**: Searches global news coverage for reporting context.
3. **Evidence Relevance Matching**: Validates retrieved articles against the claim to filter out tangential or irrelevant search results.
4. **Stance Analysis**: Maps fact-checker ratings and textual evidence into normalized stances: `SUPPORTS`, `CONTRADICTS`, or `NEUTRAL`.
5. **Auxiliary Signals**: Incorporates the V1 LinearSVC linguistic signal as secondary context.
6. **Verdict Engine**: Applies deterministic synthesis rules across evidence confidence, stance consensus, and conflict detection to output the final assessment.

---

## 4. Verdict Types & Decision Logic

The system outputs three mutually exclusive verdict categories:

| Verdict | Meaning | Decision Criteria |
|---|---|---|
| **`SUPPORTED`** | Verified True | Strong, credible evidence or authoritative fact-checker consensus confirms the claim. |
| **`CONTRADICTED`** | Verified False / Debunked | Authoritative fact-checkers or credible reporting directly refute or debunk the claim. |
| **`UNVERIFIED`** | Inconclusive / Insufficient Evidence | No relevant external evidence was found, evidence is conflicting, or available sources lack consensus. Conservative default to prevent false assertions. |

> **Role of V1 LinearSVC in V2:**
> The V1 LinearSVC model serves **only as an auxiliary linguistic signal** in V2. It does **not** determine truth by itself, cannot overrule external evidence, and is never used as the sole basis for a `SUPPORTED` or `CONTRADICTED` verdict.

---

## 5. Technology Stack

- **Backend & Pipeline**: Python 3.10+, FastAPI, Pydantic, Requests, Uvicorn
- **Machine Learning & NLP**: Scikit-Learn (LinearSVC, TfidfVectorizer), NumPy, SciPy, Pandas, Joblib
- **External APIs**:
  - Google Fact Check Tools Claim Search API
  - GDELT DOC 2.0 API (Global Database of Events, Language, and Tone)
- **Frontend**: React 19, Vite, Axios, Vanilla CSS (Glassmorphism design, responsive layouts)
- **Testing & Evaluation**: Pytest, Custom Real-World Evaluation Framework

---

## 6. Evaluation & Benchmark Results

### A. V1 In-Distribution Model Evaluation (Stylistic Classification)
- **Dataset**: WELFake Dataset (12,735 held-out test samples).
- **Held-Out Test Accuracy**: **97.38%** (Macro F1: 0.9736).
- **Important Distinction**: The 97.38% accuracy belongs strictly to the **V1 in-distribution text-classification task** on a static dataset. It is **NOT** a factual-verification accuracy metric and does not reflect real-world fact-checking capability.

### B. V2 Real-World Benchmark Evaluation (Factual Verification)
- **Benchmark Suite**: 64 real-world test cases across 8 diverse categories (established facts, viral hoaxes, breaking news, mixed claims, corporate/scientific events).
- **Latest Documented Run**:
  - **Overall Ground-Truth Accuracy**: **43.75%** (28/64 correct verdicts).
  - **False-Positive Rate**: **0.00%** (0/23) in this 64-case benchmark.
  - **Google Fact Check Hit Rate**: **20.31%** (13/64 cases returned Google Fact Check evidence).
  - **GDELT Unavailability**: **100.00%** (*GDELT was completely unavailable due to persistent HTTP 504 network timeouts during evaluation; it was not a successful retrieval source*).
  - **Conservative Defaulting**: Cases lacking external evidence safely defaulted to `UNVERIFIED`, supporting fail-safe operation under external service outages.

---

## 7. Limitations & Future Improvements

1. **Claim-Specific Evidence Retrieval**: Current keyword-based query construction can retrieve adjacent fact-checks on related topics (e.g., retrieving a viral Mars conspiracy debunk for a legitimate Mars rover discovery), leading to topic mismatch.
2. **Exact Claim-to-Source Matching**: Implementing fine-grained Natural Language Inference (NLI) or semantic entailment to rigorously verify that retrieved text directly entails or refutes the exact claim.
3. **Live News Provider Reliability**: Integrating resilient alternative news search providers (e.g., NewsAPI, Bing News, MediaCloud) to replace or augment GDELT when external endpoints experience network timeouts.
4. **Enhanced Query Formulation**: Utilizing entity-relation extraction and dependency parsing to generate more targeted search queries for complex multi-sentence claims.

---

## 8. Local Setup & Usage

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) Google Fact Check API Key for live fact-checking

### 1. Backend Setup
```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# (Optional) Set your Google Fact Check API Key in backend/.env
# GOOGLE_FACT_CHECK_API_KEY=your_key_here

# Start the FastAPI backend server
python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8008 --reload
```

### 2. Frontend Setup
```bash
# In a separate terminal
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 9. API Endpoints

- `GET /`: Service information and status.
- `GET /health`: Health check endpoint.
- `POST /predict`: V1 standalone stylistic classification endpoint.
- `POST /v2/verify`: V2 full evidence-based verification pipeline (extracts claims, retrieves external evidence, performs stance analysis, and returns final synthesized verdict).

