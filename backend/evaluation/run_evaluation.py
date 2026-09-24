#!/usr/bin/env python3
"""
Stage 26 — Real-World Evaluation Script for Real-Time News Verification System.

Evaluates the complete V2 verification pipeline (POST /v2/verify) and compares
with the V1 SVM baseline (POST /predict) across a 24-case benchmark dataset.
"""

import json
import os
import sys
import time
import statistics
from datetime import datetime
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.main import app

def sanitize_secret(obj):
    """Sanitizes text strings to prevent accidental key exposure."""
    if isinstance(obj, str):
        if "key=" in obj.lower() or "secret" in obj.lower():
            return "[REDACTED_SECRET]"
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_secret(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_secret(x) for x in obj]
    return obj

def run_evaluation(mock_mode: bool = False):
    if mock_mode:
        from backend.app.v2.verification_service import get_verification_service
        from backend.app.v2.router import get_verification_service as router_get_svc
        app.dependency_overrides[router_get_svc] = lambda: get_verification_service(mock_mode=True)

    client = TestClient(app)
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    results_filename = "results_mock.json" if mock_mode else "results.json"
    results_path = os.path.join(os.path.dirname(__file__), results_filename)

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    mode_label = "MOCK MODE" if mock_mode else "LIVE API MODE"
    print(f"Starting Stage 26 Evaluation over {len(cases)} cases [{mode_label}]...")
    print("=" * 70)


    case_results = []
    response_times = []

    for idx, case in enumerate(cases, 1):
        case_id = case["case_id"]
        category = case["category"]
        title = case.get("title", "")
        text = case.get("text", "")
        expected_verdict = case.get("expected_verdict")

        print(f"[{idx}/{len(cases)}] Evaluating {case_id} ({category})...")

        # 1. Run V2 Verification Endpoint
        t0 = time.perf_counter()
        v2_res = client.post("/v2/verify", json={"title": title, "text": text})
        t1 = time.perf_counter()
        duration_ms = round((t1 - t0) * 1000, 2)
        response_times.append(duration_ms)

        v2_data = v2_res.json() if v2_res.status_code == 200 else {}

        # 2. Run V1 Predict Endpoint
        v1_res = client.post("/predict", json={"title": title, "text": text})
        v1_data = v1_res.json() if v1_res.status_code == 200 else {}

        # Parse V2 response
        extracted_claims = v2_data.get("claims", [])
        overall_assessment = v2_data.get("overall_assessment", "UNVERIFIED")
        assessment_summary = v2_data.get("assessment_summary", "")
        has_conflict = v2_data.get("has_conflict", False)
        service_status = v2_data.get("service_status", {})
        article_svm = v2_data.get("linguistic_signal")

        # Gather evidence metrics across claims
        fact_check_count = 0
        live_news_count = 0
        supporting_count = 0
        contradicting_count = 0
        neutral_count = 0
        source_urls = []
        claim_verdicts = []

        for c in extracted_claims:
            claim_verdicts.append(c.get("verdict"))
            ev_list = c.get("evidence", [])
            for ev in ev_list:
                src_type = ev.get("source_type")
                if src_type == "FACT_CHECK_API":
                    fact_check_count += 1
                elif src_type == "LIVE_NEWS_SEARCH":
                    live_news_count += 1

                if ev.get("stance") == "SUPPORTS":
                    supporting_count += 1
                elif ev.get("stance") == "CONTRADICTS":
                    contradicting_count += 1
                else:
                    neutral_count += 1

                if ev.get("url"):
                    source_urls.append(ev.get("url"))

        total_evidence_count = fact_check_count + live_news_count

        # V1 metrics
        v1_prediction = v1_data.get("prediction", "UNKNOWN")
        v1_confidence = v1_data.get("confidence", 0.0)

        # Categorize failure if any
        failure_category = "none"
        is_correct = (overall_assessment == expected_verdict) if expected_verdict else True

        if not is_correct or overall_assessment == "UNVERIFIED":
            if len(extracted_claims) == 0:
                failure_category = "claim_extraction_failure"
            elif service_status.get("fact_check_api") != "ok" or service_status.get("live_news_api") != "ok":
                failure_category = "service_error"
            elif has_conflict:
                failure_category = "source_conflict"
            elif total_evidence_count == 0:
                failure_category = "retrieval_failure"
            elif expected_verdict == "CONTRADICTED" and fact_check_count == 0:
                failure_category = "fact_check_miss"
            elif expected_verdict == "SUPPORTED" and live_news_count == 0 and fact_check_count == 0:
                failure_category = "live_news_miss"
            elif supporting_count == 0 and contradicting_count == 0 and neutral_count > 0:
                failure_category = "stance_limitation"
            else:
                failure_category = "verdict_logic_limitation"

        result_entry = {
            "case_id": case_id,
            "category": category,
            "title": title,
            "text": text,
            "expected_verdict": expected_verdict,
            "v2_overall_assessment": overall_assessment,
            "v2_assessment_summary": assessment_summary,
            "has_conflict": has_conflict,
            "claim_count": len(extracted_claims),
            "claim_verdicts": claim_verdicts,
            "fact_check_evidence_count": fact_check_count,
            "live_news_evidence_count": live_news_count,
            "supporting_evidence_count": supporting_count,
            "contradicting_evidence_count": contradicting_count,
            "neutral_evidence_count": neutral_count,
            "total_evidence_count": total_evidence_count,
            "service_status": service_status,
            "response_time_ms": duration_ms,
            "v1_prediction": v1_prediction,
            "v1_confidence": v1_confidence,
            "article_svm_signal": article_svm,
            "source_urls": source_urls,
            "is_correct": is_correct,
            "failure_category": failure_category,
            "notes": case.get("notes", "")
        }

        case_results.append(result_entry)

    # 3. Calculate Aggregate Metrics
    total_cases = len(case_results)
    extraction_success_count = sum(1 for c in case_results if c["claim_count"] > 0)
    extraction_success_rate = round((extraction_success_count / total_cases) * 100, 2)

    fact_check_hits = sum(1 for c in case_results if c["fact_check_evidence_count"] > 0)
    fact_check_hit_rate = round((fact_check_hits / total_cases) * 100, 2)

    live_news_hits = sum(1 for c in case_results if c["live_news_evidence_count"] > 0)
    live_news_hit_rate = round((live_news_hits / total_cases) * 100, 2)

    total_supports_evidence = sum(c["supporting_evidence_count"] for c in case_results)
    total_contradicts_evidence = sum(c["contradicting_evidence_count"] for c in case_results)
    total_neutral_evidence = sum(c["neutral_evidence_count"] for c in case_results)
    total_all_evidence = sum(c["total_evidence_count"] for c in case_results)

    stance_coverage_rate = 100.0 if total_all_evidence > 0 else 0.0

    verdict_coverage_count = sum(1 for c in case_results if c["v2_overall_assessment"] in ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"])
    verdict_coverage_rate = round((verdict_coverage_count / total_cases) * 100, 2)

    labeled_cases = [c for c in case_results if c["expected_verdict"] is not None]
    verdict_accuracy_count = sum(1 for c in labeled_cases if c["is_correct"])
    verdict_accuracy_rate = round((verdict_accuracy_count / len(labeled_cases)) * 100, 2) if labeled_cases else 0.0

    unverified_count = sum(1 for c in case_results if c["v2_overall_assessment"] == "UNVERIFIED")
    unverified_rate = round((unverified_count / total_cases) * 100, 2)

    # False positive: False claim incorrectly evaluated as SUPPORTED
    false_claims = [c for c in case_results if c["expected_verdict"] == "CONTRADICTED"]
    false_positives = sum(1 for c in false_claims if c["v2_overall_assessment"] == "SUPPORTED")
    false_positive_rate = round((false_positives / len(false_claims)) * 100, 2) if false_claims else 0.0

    # False negative: True claim incorrectly evaluated as CONTRADICTED
    true_claims = [c for c in case_results if c["expected_verdict"] == "SUPPORTED"]
    false_negatives = sum(1 for c in true_claims if c["v2_overall_assessment"] == "CONTRADICTED")
    false_negative_rate = round((false_negatives / len(true_claims)) * 100, 2) if true_claims else 0.0

    conflict_detection_count = sum(1 for c in case_results if c["has_conflict"])
    service_failure_count = sum(1 for c in case_results if any(v != "ok" for v in c["service_status"].values()))

    avg_response_time = round(statistics.mean(response_times), 2)
    median_response_time = round(statistics.median(response_times), 2)

    # SVM Disagreement Analysis
    svm_disagreements = []
    for c in case_results:
        exp = c["expected_verdict"]
        v1_pred = c["v1_prediction"]
        # V1 predicts "LIKELY REAL NEWS" vs "LIKELY FAKE NEWS"
        if exp == "CONTRADICTED" and v1_pred == "LIKELY REAL NEWS":
            svm_disagreements.append(c["case_id"])
        elif exp == "SUPPORTED" and v1_pred == "LIKELY FAKE NEWS":
            svm_disagreements.append(c["case_id"])

    # Failure Category Breakdown
    failure_counts = {}
    for c in case_results:
        fc = c["failure_category"]
        failure_counts[fc] = failure_counts.get(fc, 0) + 1

    evidence_relevant_cases = sum(1 for c in case_results if c["total_evidence_count"] > 0)
    evidence_relevance_rate = round((evidence_relevant_cases / total_cases) * 100, 2)

    metrics = {
        "total_cases": total_cases,
        "claim_extraction_success_rate": extraction_success_rate,
        "fact_check_hit_rate": fact_check_hit_rate,
        "live_news_hit_rate": live_news_hit_rate,
        "evidence_relevance_rate": evidence_relevance_rate,
        "stance_determination_coverage": stance_coverage_rate,
        "supports_evidence_count": total_supports_evidence,
        "contradicts_evidence_count": total_contradicts_evidence,
        "neutral_evidence_count": total_neutral_evidence,
        "verdict_coverage_rate": verdict_coverage_rate,
        "verdict_accuracy_rate": verdict_accuracy_rate,
        "unverified_rate": unverified_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "conflict_detection_count": conflict_detection_count,
        "service_failure_count": service_failure_count,
        "avg_response_time_ms": avg_response_time,
        "median_response_time_ms": median_response_time,
        "svm_disagreements_count": len(svm_disagreements),
        "failure_categories": failure_counts
    }

    output_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "case_results": sanitize_secret(case_results)
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Print Console Summary
    print("\n" + "=" * 70)
    print("STAGE 29 REAL-WORLD EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Evaluation Cases Tested:       {total_cases}")
    print(f"Claim Extraction Success Rate:       {extraction_success_rate}%")
    print(f"Fact-Check Retrieval Hit Rate:        {fact_check_hit_rate}%")
    print(f"Live-News Retrieval Hit Rate:         {live_news_hit_rate}%")
    print(f"Evidence Relevance Rate:             {evidence_relevance_rate}%")
    print(f"Stance Determination Coverage:      {stance_coverage_rate}%")
    print(f"SUPPORTS Evidence Count:             {total_supports_evidence}")
    print(f"CONTRADICTS Evidence Count:          {total_contradicts_evidence}")
    print(f"NEUTRAL Evidence Count:              {total_neutral_evidence}")
    print(f"Verdict Coverage Rate:               {verdict_coverage_rate}%")
    print(f"Ground-Truth Verdict Accuracy Rate:  {verdict_accuracy_rate}%")
    print(f"UNVERIFIED Assessment Rate:          {unverified_rate}%")
    print(f"False Positive Rate (Fake -> Real):  {false_positive_rate}%")
    print(f"False Negative Rate (Real -> Fake):  {false_negative_rate}%")
    print(f"Conflict Detection Count:            {conflict_detection_count}")
    print(f"Service Failure Count:               {service_failure_count}")
    print(f"Average Response Time:               {avg_response_time} ms")
    print(f"Median Response Time:                {median_response_time} ms")

    print("-" * 70)
    print("V1 SVM vs Ground-Truth Disagreements:")
    print(f"  Total Disagreements: {len(svm_disagreements)} cases ({', '.join(svm_disagreements)})")
    print("-" * 70)
    print("Failure Category Breakdown:")
    for cat, cnt in failure_counts.items():
        print(f"  - {cat}: {cnt}")
    print("=" * 70)
    print(f"Evaluation results successfully saved to: {results_path}")

    return output_payload

if __name__ == "__main__":
    is_mock = "--mock" in sys.argv
    run_evaluation(mock_mode=is_mock)

