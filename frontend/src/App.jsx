import React, { useState } from "react";
import NewsForm from "./components/NewsForm";
import PredictionResult from "./components/PredictionResult";
import { predictNews } from "./api";

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [validationError, setValidationError] = useState("");

  const handlePredict = async ({ title, text }) => {
    setIsLoading(true);
    setError("");
    setResult(null); // Clear previous result while loading

    try {
      const data = await predictNews(title, text);
      setResult(data);
    } catch (err) {
      setError(err.message || "An error occurred while analyzing the news content.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="main-title">Fake News Detector</h1>
        <p className="subtitle">
          Analyze a news article using a machine learning classification model.
        </p>
      </header>

      <main className="main-content">
        <div className="card form-card">
          <NewsForm
            onSubmit={handlePredict}
            isLoading={isLoading}
            validationError={validationError}
            setValidationError={setValidationError}
          />
        </div>

        {error && (
          <div className="error-card" role="alert">
            <div className="error-title">Error</div>
            <div className="error-message">{error}</div>
          </div>
        )}

        {result && (
          <div className="card result-container">
            <PredictionResult result={result} />
          </div>
        )}
      </main>

      <footer className="footer">
        <div className="disclaimer-container">
          <p className="disclaimer-text">
            <strong>Disclaimer:</strong> This system predicts patterns associated with fake or real news examples in its training data. It does not independently verify facts or guarantee that an article is true or false.
          </p>
          <p className="disclaimer-text">
            Model confidence is derived from an SVM decision margin and is not a calibrated probability.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
