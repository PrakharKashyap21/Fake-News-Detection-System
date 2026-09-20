# Fake News Detection System using Machine Learning & NLP

A production-grade Machine Learning and Natural Language Processing (NLP) system designed to analyze news headlines and body text to predict whether an article resembles real or fake news based on statistical patterns learned from the WELFake dataset.

---

## Architecture & System Flow

```
Browser  ──►  React UI  ──►  Axios  ──►  FastAPI Backend  ──►  TF-IDF + Linear SVM  ──►  JSON Response  ──►  React Result Card
```

---

## Features

- **NLP Pipeline**: Custom text normalization and TF-IDF feature extraction (unigrams + bigrams).
- **Leakage-Safe Methodology**: Stratified train/test splitting and TF-IDF fitting strictly isolated to training data.
- **Optimized Machine Learning**: Linear Support Vector Machine (LinearSVC) tuned via 12 controlled hyperparameter and feature extraction experiments.
- **FastAPI Backend API**: Lightweight, RESTful API providing real-time text classification and model confidence scoring.
- **Modern React Frontend**: Clean, responsive single-page web interface built with React, Vite, Axios, and Vanilla CSS.

---

## Technology Stack

- **Core & Language**: Python 3.10+, JavaScript (ES6+)
- **Machine Learning & NLP**: Scikit-Learn, SciPy, NumPy, Pandas, Joblib
- **Feature Extraction**: TF-IDF Vectorizer (1,016,445 n-grams)
- **Classifier**: Linear Support Vector Machine (`LinearSVC`, `C=2.0`)
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend UI**: React 19, Vite, Axios, Vanilla CSS

---

## Dataset & Preprocessing

- **Dataset**: WELFake Dataset (72,134 raw news articles: 35,028 FAKE, 37,106 REAL).
- **Data Cleaning**:
  - Dropped index column (`Unnamed: 0`).
  - Normalized title and body text (safe string conversion, whitespace trimming, space folding).
  - Merged title and text into a unified `content` feature.
  - Removed 8,462 redundant exact duplicate content rows prior to splitting.
  - Final Cleaned Dataset Size: **63,672 articles** (34,789 FAKE / 54.64%, 28,883 REAL / 45.36%).
- **Train / Test Split**:
  - Stratified 80% Training (**50,937 samples**) / 20% Held-Out Testing (**12,735 samples**).
  - Verified **0 overlapping normalized content items** between train and test sets to guarantee no data leakage.

---

## Model Selection & Optimization

1. **Model Comparison (80/20 Inner Validation Split)**:
   - Logistic Regression (`C=1.0`): Macro F1 = 0.9515
   - Multinomial Naive Bayes (`alpha=1.0`): Macro F1 = 0.8424
   - **Linear Support Vector Machine (`C=1.0`)**: Macro F1 = **0.9715** (Top Performer)

2. **Hyperparameter & TF-IDF Tuning (12 Controlled Experiments)**:
   - Best Configuration: `Config C` (`ngram_range=(1, 2)`, `min_df=3`, `max_df=0.95`, `sublinear_tf=True`) paired with `LinearSVC(C=2.0, max_iter=5000, random_state=42)`.
   - Validation Macro F1: **0.9733** | Validation Accuracy: **97.35%**.

---

## Final Production Model Performance (Held-Out Test Set)

Evaluated **exactly once** on the locked test set (**12,735 samples**):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97.38%** (0.9738) |
| **Macro F1-Score** | **0.9736** |
| **Weighted F1-Score** | **0.9738** |

### Per-Class Detailed Performance

| Class | Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FAKE** | 0 | **97.82%** | **97.37%** | **0.9759** | 6,958 |
| **REAL** | 1 | **96.85%** | **97.39%** | **0.9712** | 5,777 |

### Confusion Matrix

```
                  Predicted
                  FAKE      REAL
Actual FAKE       6775       183
Actual REAL        151      5626
```
- **True Negatives (TN - Actual FAKE, Pred FAKE)**: 6,775
- **False Positives (FP - Actual FAKE, Pred REAL)**: 183
- **False Negatives (FN - Actual REAL, Pred FAKE)**: 151
- **True Positives (TP - Actual REAL, Pred REAL)**: 5,626

---

## Local Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher & npm

### 1. Backend Setup
From the project root directory:

```bash
# Install backend Python dependencies
pip install -r backend/requirements.txt

# Start the FastAPI backend server on port 8008
python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8008 --reload
```

The API will be live at `http://127.0.0.1:8008`.

### 2. Frontend Setup
Open a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the React Vite dev server
npm run dev
```

Open `http://localhost:5173` in your browser to use the single-page application.

---

## API Documentation

### `POST /predict`
Classifies a news article headline and/or body text.

#### Example Request
```json
POST http://127.0.0.1:8008/predict
Content-Type: application/json

{
  "title": "Federal Reserve Maintained Benchmark Interest Rates",
  "text": "The Federal Reserve held interest rates steady today following a two-day meeting in Washington."
}
```

#### Example Response (HTTP 200 OK)
```json
{
  "prediction": "LIKELY REAL NEWS",
  "label": 1,
  "confidence": 95.31,
  "message": "The model detected linguistic patterns associated with real-news examples in its training data."
}
```

---

## Project Disclaimers & Limitations

- **Statistical Pattern Recognition**: The model predicts classification based exclusively on statistical linguistic patterns learned from the WELFake training dataset.
- **No Factual Verification**: A prediction of `LIKELY REAL NEWS` does **not** guarantee objective factual truth.
- **No Independent Fact-Checking**: A prediction of `LIKELY FAKE NEWS` does **not** constitute independent fact-checking or journalistic verification.
- **Model Confidence Score**: The `confidence` value is a transformed SVM decision-margin score ($100 \times (1 - e^{-|\text{margin}|})$) and is **not** a calibrated probability.
- **Local Application**: Cloud deployment has not been performed; the system runs strictly in your local environment.
