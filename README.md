# Fake News Detection System

A machine-learning-based text classification application that predicts whether submitted news content resembles FAKE or REAL news based on statistical patterns learned from the WELFake dataset.

---

## Overview

The Fake News Detection System provides a streamlined machine learning pipeline and single-page web interface to analyze news headlines and body text. The system leverages Natural Language Processing (NLP) techniques and a Linear Support Vector Machine (LinearSVC) classifier to detect linguistic patterns associated with fake or real news.

> **Note:** This application is a statistical pattern recognition system and is not an authoritative fact-checker or truth detector.

---

## Architecture

```
Browser
  ↓
React + Vite
  ↓
Axios
  ↓
FastAPI
  ↓
TF-IDF Vectorizer
  ↓
LinearSVC Model
  ↓
JSON response
  ↓
React result card
```

---

## Features

- **Headline & Article Classification**: Accepts optional headlines and article body text for prediction.
- **TF-IDF Text Representation**: Extracts unigram and bigram features with sublinear term frequency scaling.
- **Linear SVM Classification**: High-accuracy binary classification model optimized for high-dimensional sparse text data.
- **FastAPI Backend API**: High-performance RESTful API endpoints with request validation and health monitoring.
- **React Frontend**: Clean, responsive single-page user interface with real-time model confidence visualization.
- **Decision Margin Confidence**: Derived directly from the SVM decision margin for model output interpretation.

---

## Technology Stack

- **Backend / Machine Learning**: Python 3.10+, Scikit-Learn, SciPy, NumPy, Pandas, Joblib, FastAPI, Uvicorn, Pydantic
- **Frontend**: React 19, Vite, Axios, Vanilla CSS
- **Dataset**: WELFake Dataset (Kaggle)

---

## Dataset & Preprocessing

- **Dataset**: WELFake Dataset (72,134 raw news records).
- **Data Cleaning**:
  - Removed non-content index column (`Unnamed: 0`).
  - Safe text normalization (string conversion, whitespace trimming, and space folding).
  - Combined headline and article body into a single `content` feature.
  - Validated content and label availability (labels: `0 = FAKE`, `1 = REAL`).
  - Removed 8,462 redundant exact duplicate content records prior to train/test split.
  - Final Cleaned Dataset: **63,672 records** (34,789 FAKE / 54.64%, 28,883 REAL / 45.36%).

---

## Data Split

- **Training Samples**: 50,937 samples (80%)
- **Held-Out Test Samples**: 12,735 samples (20%)
- **Split Strategy**: Stratified 80/20 train/test split (`random_state=42`).
- **Data Leakage Validation**: **0 overlapping normalized content records** between training and testing sets.

---

## Machine Learning Pipeline

- **Text Normalization**: Lowercased, unicode accent stripping, whitespace folded.
- **TF-IDF Configuration**:
  - `TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)`
  - Vocabulary Size: **1,016,445 n-grams** (learned exclusively from training data).
- **Classifier Configuration**:
  - `LinearSVC(C=2.0, max_iter=5000, random_state=42)`

---

## Model Selection & Optimization

Model selection was conducted using **12 controlled hyperparameter and feature extraction experiments** on an internal 80/20 stratified validation split created exclusively from the training dataset.

Candidate models evaluated during selection:
- **Logistic Regression** (`C=1.0`): Validation Macro F1 = 0.9515
- **Multinomial Naive Bayes** (`alpha=1.0`): Validation Macro F1 = 0.8424
- **Linear Support Vector Machine** (`C=1.0`): Validation Macro F1 = 0.9715

Following hyperparameter tuning across C parameters (0.5, 1.0, 2.0) and TF-IDF configurations, `LinearSVC` with `C=2.0` and `min_df=3` unigram+bigram TF-IDF yielded the top validation performance (Validation Macro F1: **0.9733**).

> **Methodology Note:** The held-out test set was used only for final evaluation and was not used for model selection.

---

## Final Production Model Results

The final production model was evaluated **exactly once** on the locked held-out test set (**12,735 samples**):

### Overall Held-Out Test Metrics
- **Accuracy**: **97.38%** (0.9738)
- **Macro F1-Score**: **0.9736**
- **Weighted F1-Score**: **0.9738**

### Per-Class Performance Breakdown

| Class | Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FAKE** | 0 | **97.82%** | **97.37%** | **0.9759** | 6,958 |
| **REAL** | 1 | **96.85%** | **97.39%** | **0.9712** | 5,777 |

### Confusion Matrix

```
             Predicted
             FAKE   REAL

Actual FAKE  6775    183
Actual REAL   151   5626
```

- **True Negatives (Actual FAKE, Predicted FAKE)**: 6,775
- **False Positives (Actual FAKE, Predicted REAL)**: 183
- **False Negatives (Actual REAL, Predicted FAKE)**: 151
- **True Positives (Actual REAL, Predicted REAL)**: 5,626

> The held-out test set was used only for final evaluation and was not used for model selection.

---

## API Documentation

### Endpoints

#### `GET /`
Service information endpoint.
```json
{
  "status": "ok",
  "service": "Fake News Detection API"
}
```

#### `GET /health`
Health check status.
```json
{
  "status": "healthy"
}
```

#### `POST /predict`
Classifies news headline and/or article text.

**Example Request:**
```json
{
  "title": "U.S. Federal Reserve Announces Interest Rate Decision",
  "text": "The Federal Reserve kept interest rates unchanged today following its two-day policy meeting in Washington."
}
```

**Example Response (HTTP 200 OK):**
```json
{
  "prediction": "LIKELY REAL NEWS",
  "label": 1,
  "confidence": 95.31,
  "message": "The model detected linguistic patterns associated with real-news examples in its training data."
}
```

### Model Confidence Calculation
Model confidence is calculated from the SVM decision margin using a deterministic transformation:
$$\text{confidence} = 100 \times (1 - e^{-|\text{decision\_margin}|})$$
This value represents a transformed decision-margin score and is **not** a calibrated probability.

---

## Local Setup & Run Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Backend Setup
From the project root:

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Start the FastAPI backend server
python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8008 --reload
```

### 2. Frontend Setup
In a new terminal window:

```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start the React Vite dev server
npm run dev
```

Open `http://localhost:5173` in your browser.

> **Note:** Production model artifacts (`backend/models/*.joblib`) and datasets (`backend/data/*.csv`) are intentionally excluded from Git tracking via `.gitignore` and must exist locally for the application workflow.

---

## Limitations & Disclaimers

- **Dataset Dependence**: Model predictions reflect statistical linguistic patterns learned from the WELFake dataset and may not generalize to all news domains or emerging topics.
- **Statistical Prediction**: The model predicts pattern similarity and does **not** independently verify facts, perform journalistic investigation, or guarantee objective truth.
- **Error Margin**: Like all machine learning models, predictions can be incorrect.
- **Uncalibrated Confidence Score**: Model confidence is derived from an SVM decision margin and is **not** a calibrated probability.
- **Dataset Coverage**: The WELFake dataset contains historical news data and may not capture current events or evolving misinformation formats.

---

## Future Deployment

Cloud deployment (e.g., Render for FastAPI backend, Vercel for React frontend, or containerization with Docker) can be added in future iterations. Currently, the application is configured for local production-like simulation.
