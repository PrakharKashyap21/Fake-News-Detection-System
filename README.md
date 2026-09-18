# Fake-News-Detection-System
A machine learning and NLP based system for detecting potentially fake news articles.

## Project Status
Backend classification pipeline and FastAPI prediction endpoints are fully implemented and evaluated.

### Planned Features
- [x] News text preprocessing using NLP
- [x] TF-IDF feature extraction
- [x] Machine Learning based classification (Linear SVM)
- [x] Model evaluation with accuracy, precision, recall and F1-score
- [x] FastAPI backend
- [ ] React + Vite frontend
- [ ] Real-time news classification interface

### Technology Stack
- Python
- Pandas & NumPy
- Scikit-learn & SciPy
- TF-IDF Vectorizer
- Linear SVM Classifier
- FastAPI & Uvicorn
- Pydantic

---

## API Usage

### Endpoints

#### `GET /`
Service status check.
```json
{
  "status": "ok",
  "service": "Fake News Detection API"
}
```

#### `GET /health`
Health check endpoint.
```json
{
  "status": "healthy"
}
```

#### `POST /predict`
Classifies a news article headline and/or body text.

**Example Request:**
```json
{
  "title": "U.S. Federal Reserve Announces Benchmark Interest Rate Decision",
  "text": "The Federal Reserve kept interest rates unchanged today following its two-day policy meeting in Washington."
}
```

**Example Response:**
```json
{
  "prediction": "LIKELY REAL NEWS",
  "label": 1,
  "confidence": 95.31,
  "message": "The model detected linguistic patterns associated with real-news examples in its training data."
}
```

---

## Disclaimers & Methodology
- **Training Data Patterns**: The model predicts based on statistical linguistic patterns learned from the WELFake training dataset.
- **No Factual Verification**: "LIKELY REAL NEWS" does **not** guarantee objective factual truth.
- **No Independent Fact-Checking**: "LIKELY FAKE NEWS" indicates statistical similarity to training examples and does **not** constitute independent fact-checking or journalistic verification.
- **Confidence Score**: The `confidence` value is a transformed SVM decision-margin score (`100 * (1 - exp(-|margin|))`) and is **not** a calibrated probability.
