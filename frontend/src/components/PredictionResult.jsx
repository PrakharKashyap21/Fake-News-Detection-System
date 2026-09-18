import React from "react";

const PredictionResult = ({ result }) => {
  if (!result) return null;

  const { prediction, label, confidence, message } = result;

  const isReal = label === 1;

  return (
    <div className={`result-card ${isReal ? "result-real" : "result-fake"}`}>
      <div className="result-header">
        <div className={`prediction-badge ${isReal ? "badge-real" : "badge-fake"}`}>
          {prediction}
        </div>
        <div className="confidence-tag">
          Model confidence: <strong>{confidence.toFixed(2)}%</strong>
        </div>
      </div>

      <p className="result-message">{message}</p>

      <div className="result-note">
        Confidence is derived from the model's SVM decision margin and is not a calibrated probability.
      </div>
    </div>
  );
};

export default PredictionResult;
