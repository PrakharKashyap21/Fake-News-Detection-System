import React, { useState } from "react";
import NewsForm from "./components/NewsForm";
import PredictionResult from "./components/PredictionResult";
import V2VerificationResult from "./components/V2VerificationResult";
import { predictNews, verifyNewsV2 } from "./api";

function App() {
  const [activeTab, setActiveTab] = useState("v2"); // 'v2' (default) or 'v1'
  const [isLoading, setIsLoading] = useState(false);
  const [v1Result, setV1Result] = useState(null);
  const [v2Result, setV2Result] = useState(null);
  const [error, setError] = useState("");
  const [validationError, setValidationError] = useState("");

  const handleTabChange = (tab) => {
    if (isLoading) return;
    setActiveTab(tab);
    setError("");
    setValidationError("");
  };

  const handleVerifyV2 = async ({ title, text }) => {
    setIsLoading(true);
    setError("");
    setV2Result(null);

    try {
      const data = await verifyNewsV2(title, text);
      setV2Result(data);
    } catch (err) {
      setError(err.message || "An error occurred while performing real-time verification.");
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredictV1 = async ({ title, text }) => {
    setIsLoading(true);
    setError("");
    setV1Result(null);

    try {
      const data = await predictNews(title, text);
      setV1Result(data);
    } catch (err) {
      setError(err.message || "An error occurred while analyzing the news content.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="main-title">Real-Time News Verification System</h1>
        <p className="subtitle">
          Fact-checking and live news evidence aggregation pipeline.
        </p>

        {/* Mode Navigation Tabs */}
        <div className="tab-navigation" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "v2"}
            className={`tab-btn ${activeTab === "v2" ? "active" : ""}`}
            onClick={() => handleTabChange("v2")}
            disabled={isLoading}
          >
            V2 Real-Time Verification
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "v1"}
            className={`tab-btn ${activeTab === "v1" ? "active" : ""}`}
            onClick={() => handleTabChange("v1")}
            disabled={isLoading}
          >
            V1 Pattern Classifier (SVM Baseline)
          </button>
        </div>
      </header>

      <main className="main-content">
        <div className="card form-card">
          {activeTab === "v2" ? (
            <NewsForm
              key="v2-form"
              onSubmit={handleVerifyV2}
              isLoading={isLoading}
              validationError={validationError}
              setValidationError={setValidationError}
              submitLabel="Verify News"
            />
          ) : (
            <NewsForm
              key="v1-form"
              onSubmit={handlePredictV1}
              isLoading={isLoading}
              validationError={validationError}
              setValidationError={setValidationError}
              submitLabel="Detect News"
            />
          )}
        </div>

        {error && (
          <div className="error-card" role="alert">
            <div className="error-title">Error</div>
            <div className="error-message">{error}</div>
          </div>
        )}

        {/* V2 Real-Time Verification Output */}
        {activeTab === "v2" && v2Result && (
          <div className="v2-result-wrapper">
            <V2VerificationResult result={v2Result} />
          </div>
        )}

        {/* V1 SVM Baseline Output */}
        {activeTab === "v1" && v1Result && (
          <div className="card result-container">
            <PredictionResult result={v1Result} />
          </div>
        )}
      </main>

      <footer className="footer">
        <div className="disclaimer-container">
          <p className="disclaimer-text">
            <strong>Disclaimer:</strong> V2 Verification aggregates live news and Google Fact Check API reporting. Claims marked UNVERIFIED indicate an absence of matched external evidence rather than proven falsity.
          </p>
          <p className="disclaimer-text">
            V1 SVM model outputs represent linguistic pattern matching against historical training data and are strictly baseline indicators.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
