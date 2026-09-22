import React, { useState } from "react";

const V2VerificationResult = ({ result }) => {
  if (!result) return null;

  const {
    overall_assessment,
    assessment_summary,
    has_conflict,
    claims,
    linguistic_signal,
    service_status,
    disclaimer
  } = result;

  const getVerdictClass = (verdict) => {
    switch (verdict) {
      case "SUPPORTED":
        return "badge-supported";
      case "CONTRADICTED":
        return "badge-contradicted";
      case "UNVERIFIED":
      default:
        return "badge-unverified";
    }
  };

  const getStrengthClass = (strength) => {
    switch (strength) {
      case "STRONG":
        return "strength-strong";
      case "MODERATE":
        return "strength-moderate";
      case "LIMITED":
        return "strength-limited";
      case "NONE":
      default:
        return "strength-none";
    }
  };

  const getUncertaintyClass = (level) => {
    switch (level) {
      case "LOW":
        return "uncertainty-low";
      case "MEDIUM":
        return "uncertainty-medium";
      case "HIGH":
      default:
        return "uncertainty-high";
    }
  };

  const getStanceBadge = (stance) => {
    switch (stance) {
      case "SUPPORTS":
        return <span className="stance-badge stance-supports">Supports</span>;
      case "CONTRADICTS":
        return <span className="stance-badge stance-contradicts">Contradicts</span>;
      case "NEUTRAL":
      default:
        return <span className="stance-badge stance-neutral">Neutral</span>;
    }
  };

  return (
    <div className="v2-verification-container">
      {/* Overall Assessment Header */}
      <div className={`overall-card ${getVerdictClass(overall_assessment)}`}>
        <div className="overall-header">
          <div className="overall-title-group">
            <span className="overall-label">Overall Assessment</span>
            <span className={`verdict-pill ${getVerdictClass(overall_assessment)}`}>
              {overall_assessment}
            </span>
          </div>
          {has_conflict && (
            <span className="conflict-flag-badge">
              ⚠️ Conflicting Evidence Detected
            </span>
          )}
        </div>
        <p className="overall-summary">{assessment_summary}</p>
      </div>

      {/* Service Status Warning Banners */}
      {service_status && (service_status.fact_check_api !== "ok" || service_status.live_news_api !== "ok") && (
        <div className="service-warning-banner" role="alert">
          <div className="service-warning-title">⚠️ Source Availability Notice</div>
          <ul>
            {service_status.fact_check_api === "missing_api_key" && (
              <li>Google Fact Check API key is not configured on the server. Fact-check evidence retrieval is currently unavailable.</li>
            )}
            {service_status.fact_check_api && service_status.fact_check_api.startsWith("error_") && (
              <li>Google Fact Check API encountered an error ({service_status.fact_check_api}). Fact-check results may be incomplete.</li>
            )}
            {service_status.live_news_api && service_status.live_news_api !== "ok" && (
              <li>Live news search service encountered an error ({service_status.live_news_api}). News coverage retrieval may be incomplete.</li>
            )}
          </ul>
        </div>
      )}

      {/* Article-level Linguistic SVM Signal Banner */}
      {linguistic_signal && (
        <div className="linguistic-signal-card">
          <div className="linguistic-signal-header">
            <span className="signal-title">Linguistic Pattern Signal (V1 SVM Baseline)</span>
            <span className="signal-score">Margin Confidence: {linguistic_signal.confidence.toFixed(1)}%</span>
          </div>
          <p className="signal-message">{linguistic_signal.message}</p>
          <div className="signal-disclaimer">
            Note: The linguistic pattern signal is derived strictly from text style classification (V1 SVM) and does NOT evaluate factual truth or external evidence.
          </div>
        </div>
      )}

      {/* Extracted Claims Section */}
      <div className="claims-section">
        <h2 className="claims-section-title">Claim-by-Claim Verification ({claims ? claims.length : 0})</h2>

        {!claims || claims.length === 0 ? (
          <div className="empty-claims-card">
            No distinct factual claims were extracted from the input text.
          </div>
        ) : (
          claims.map((claim, index) => (
            <ClaimCard
              key={claim.claim_id || index}
              claim={claim}
              index={index}
              getVerdictClass={getVerdictClass}
              getStrengthClass={getStrengthClass}
              getUncertaintyClass={getUncertaintyClass}
              getStanceBadge={getStanceBadge}
            />
          ))
        )}
      </div>

      {/* System Disclaimer */}
      {disclaimer && (
        <div className="v2-disclaimer-card">
          <p>{disclaimer}</p>
        </div>
      )}
    </div>
  );
};

const ClaimCard = ({ claim, index, getVerdictClass, getStrengthClass, getUncertaintyClass, getStanceBadge }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const {
    text,
    verdict,
    reasoning,
    evidence_strength,
    uncertainty_level,
    has_conflicting_evidence,
    supporting_evidence_count,
    contradicting_evidence_count,
    neutral_evidence_count,
    evidence,
    evidence_summary,
    linguistic_signal
  } = claim;

  const fcEvidence = evidence ? evidence.filter(e => e.source_type === "FACT_CHECK_API") : [];
  const newsEvidence = evidence ? evidence.filter(e => e.source_type === "LIVE_NEWS_SEARCH") : [];
  const totalEvidenceCount = evidence ? evidence.length : 0;
  const isLiveNewsOnly = fcEvidence.length === 0 && newsEvidence.length > 0;

  return (
    <div className="claim-card">
      <div className="claim-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="claim-title-area">
          <span className="claim-number">Claim #{index + 1}</span>
          <p className="claim-text">"{text}"</p>
        </div>

        <div className="claim-meta-tags">
          <span className={`verdict-pill ${getVerdictClass(verdict)}`}>{verdict}</span>
          <span className={`meta-pill ${getStrengthClass(evidence_strength)}`}>
            Strength: {evidence_strength}
          </span>
          <span className={`meta-pill ${getUncertaintyClass(uncertainty_level)}`}>
            Uncertainty: {uncertainty_level}
          </span>
          <button type="button" className="expand-toggle-btn" aria-label="Toggle claim details">
            {isExpanded ? "▲" : "▼"}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="claim-body">
          {/* Deterministic Reasoning */}
          <div className="reasoning-box">
            <strong>Verification Reasoning:</strong> {reasoning}
          </div>

          {/* Special Situation Banners */}
          {has_conflicting_evidence && (
            <div className="claim-alert alert-conflict">
              <strong>Conflicting evidence found:</strong> Supporting and contradicting sources were retrieved for this claim. Review individual evidence items below.
            </div>
          )}

          {isLiveNewsOnly && (
            <div className="claim-alert alert-livenews-context">
              <strong>Live news coverage notice:</strong> Only live-news coverage was retrieved for this claim. News coverage provides context and indicates active media reporting, but is NOT proof of factual truth.
            </div>
          )}

          {totalEvidenceCount === 0 && (
            <div className="claim-alert alert-no-evidence">
              <strong>No supporting or contradicting evidence found.</strong> No independent fact-checks or live news coverage were matched for this claim.
            </div>
          )}

          {/* Per-Claim Linguistic Signal */}
          {linguistic_signal && (
            <div className="claim-linguistic-signal">
              <span className="signal-badge-label">Linguistic Signal (V1 SVM):</span>
              <span className="signal-badge-val">{linguistic_signal.prediction} ({linguistic_signal.confidence.toFixed(1)}% margin)</span>
            </div>
          )}

          {/* Evidence Items Section */}
          {totalEvidenceCount > 0 && (
            <div className="evidence-sections-wrapper">
              <h4 className="evidence-heading">
                Retrieved Evidence ({totalEvidenceCount} items: {fcEvidence.length} Fact-Checks, {newsEvidence.length} Live News)
              </h4>

              {/* Fact Check Evidence List */}
              {fcEvidence.length > 0 && (
                <div className="evidence-group">
                  <h5 className="group-title">🔍 Fact-Check Reports ({fcEvidence.length})</h5>
                  <div className="evidence-grid">
                    {fcEvidence.map((item, idx) => (
                      <EvidenceCard key={item.id || idx} item={item} getStanceBadge={getStanceBadge} />
                    ))}
                  </div>
                </div>
              )}

              {/* Live News Evidence List */}
              {newsEvidence.length > 0 && (
                <div className="evidence-group">
                  <h5 className="group-title">📰 Live News Coverage ({newsEvidence.length})</h5>
                  <div className="evidence-grid">
                    {newsEvidence.map((item, idx) => (
                      <EvidenceCard key={item.id || idx} item={item} getStanceBadge={getStanceBadge} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const EvidenceCard = ({ item, getStanceBadge }) => {
  return (
    <div className="evidence-item-card">
      <div className="evidence-item-header">
        <div className="publisher-info">
          <span className="publisher-name">{item.publisher || item.domain}</span>
          {item.publish_date && <span className="publish-date">• {item.publish_date}</span>}
        </div>
        {getStanceBadge(item.stance)}
      </div>

      <h5 className="evidence-title">
        <a href={item.url} target="_blank" rel="noopener noreferrer" className="evidence-link">
          {item.title} ↗
        </a>
      </h5>

      {item.snippet && <p className="evidence-snippet">"{item.snippet}"</p>}

      <div className="evidence-footer">
        <span className="source-type-tag">
          {item.source_type === "FACT_CHECK_API" ? "Fact Check API" : "Live News Search"}
        </span>
        {item.raw_rating && (
          <span className="raw-rating-tag">
            Raw Rating: <strong>{item.raw_rating}</strong>
          </span>
        )}
      </div>
    </div>
  );
};

export default V2VerificationResult;
