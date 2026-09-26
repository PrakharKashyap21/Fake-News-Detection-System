#!/usr/bin/env python3
"""
Stage 30 — Real-World API Evaluation & Benchmark Script
Real-Time News Verification System

Evaluates the complete V2 verification pipeline (POST /v2/verify) using real external
retrieval (Google Fact Check API and GDELT DOC 2.0 API) against an expanded 64-case dataset.
"""

import json
import os
import sys
import time
import math
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.main import app
from backend.app.v2.schemas import StanceType, EvidenceSourceType
from backend.app.v2.newsapi_retriever import load_env_key

def sanitize_secret(obj: Any) -> Any:
    """Sanitizes text strings to prevent accidental API key exposure."""
    if isinstance(obj, str):
        if "key=" in obj.lower() or "secret" in obj.lower():
            # Redact query string keys
            import re
            cleaned = re.sub(r'key=[^&]+', 'key=[REDACTED_KEY]', obj)
            return cleaned
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_secret(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_secret(x) for x in obj]
    return obj


def calculate_p95(values: List[float]) -> float:
    """Calculates 95th percentile value from a list of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(math.ceil(0.95 * len(sorted_vals))) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def classify_failure(
    case: Dict[str, Any],
    v2_assessment: str,
    ground_truth: str,
    extracted_claims: List[Any],
    fact_check_count: int,
    live_news_count: int,
    total_evidence_count: int,
    relevant_count: int,
    supporting_count: int,
    contradicting_count: int,
    neutral_count: int,
    has_conflict: bool,
    service_status: Dict[str, str]
) -> str:
    """
    Classifies errors into one of 10 standard failure categories:
    - claim_extraction_failure
    - fact_check_retrieval_failure
    - live_news_retrieval_failure
    - irrelevant_evidence
    - stance_error
    - verdict_logic_error
    - source_conflict
    - insufficient_evidence
    - ground_truth_ambiguity
    - service_failure
    """
    is_correct = (v2_assessment == ground_truth)
    if is_correct:
        return "none"

    # 1. External Service Failure
    if any(status_val != "ok" for status_val in service_status.values()):
        return "service_failure"

    # 2. Claim Extraction Failure
    if len(extracted_claims) == 0:
        return "claim_extraction_failure"

    # 3. Source Conflict
    if has_conflict:
        return "source_conflict"

    # 4. Retrieval Failures
    if ground_truth == "CONTRADICTED" and fact_check_count == 0 and live_news_count == 0:
        return "fact_check_retrieval_failure"
    elif ground_truth == "SUPPORTED" and live_news_count == 0 and fact_check_count == 0:
        return "live_news_retrieval_failure"
    elif total_evidence_count == 0 and ground_truth in ("SUPPORTED", "CONTRADICTED"):
        return "insufficient_evidence"

    # 5. Irrelevant Evidence
    if total_evidence_count > 0 and relevant_count == 0:
        return "irrelevant_evidence"

    # 6. Stance Error
    if (ground_truth == "CONTRADICTED" and supporting_count > 0 and contradicting_count == 0) or \
       (ground_truth == "SUPPORTED" and contradicting_count > 0 and supporting_count == 0):
        return "stance_error"

    # 7. Ground Truth Ambiguity
    if ground_truth == "UNVERIFIED" and v2_assessment != "UNVERIFIED":
        return "ground_truth_ambiguity"

    # 8. Verdict Logic Error
    return "verdict_logic_error"


def run_real_world_evaluation(
    mock_mode: bool = False,
    start_idx: int = 0,
    limit: Optional[int] = None,
    output_file: Optional[str] = None
):
    # 1. Check API Configuration
    fc_key = os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "").strip() or load_env_key("GOOGLE_FACT_CHECK_API_KEY")
    if fc_key and not os.environ.get("GOOGLE_FACT_CHECK_API_KEY"):
        os.environ["GOOGLE_FACT_CHECK_API_KEY"] = fc_key
    fc_api_status = "configured" if fc_key else "missing"

    news_key = os.environ.get("NEWS_API_KEY", "").strip() or load_env_key("NEWS_API_KEY")
    if news_key and not os.environ.get("NEWS_API_KEY"):
        os.environ["NEWS_API_KEY"] = news_key
    news_api_status = "configured" if news_key else "missing"

    if mock_mode:
        from backend.app.v2.verification_service import get_verification_service
        from backend.app.v2.router import get_verification_service as router_get_svc
        app.dependency_overrides[router_get_svc] = lambda: get_verification_service(mock_mode=True)
        eval_mode_title = "MOCK BENCHMARK EVALUATION (OFFLINE)"
    else:
        eval_mode_title = "COMPLETE REAL-WORLD API EVALUATION" if (fc_api_status == "configured" and news_api_status == "configured") else "PARTIAL REAL-WORLD API EVALUATION"

    client = TestClient(app)
    dataset_path = os.path.join(os.path.dirname(__file__), "real_world_dataset.json")
    results_path = output_file or os.path.join(os.path.dirname(__file__), "real_world_results.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    cases = all_cases[start_idx : start_idx + limit] if limit is not None else all_cases[start_idx:]

    print("=" * 80)
    print(f"STAGE 32 — REAL-WORLD EVALUATION [{eval_mode_title}]")
    print(f"Dataset Size: {len(cases)} benchmark cases (from total {len(all_cases)})")
    print(f"Google Fact Check API Key: {fc_api_status}")
    print(f"NewsAPI Key: {news_api_status}")
    print("=" * 80)

    case_results = []
    response_times = []

    for idx, case in enumerate(cases, 1):
        case_id = case["id"]
        category = case["category"]
        title = case.get("title", "")
        text = case.get("text", "")
        ground_truth = case.get("ground_truth")
        ground_truth_source = case.get("ground_truth_source", "")
        source_date = case.get("source_date", "")

        print(f"[{idx:02d}/{len(cases):02d}] Evaluating {case_id} ({category[:25]}...)...")

        # Execute V2 Verification API call
        t0 = time.perf_counter()
        v2_res = client.post("/v2/verify", json={"title": title, "text": text})
        t1 = time.perf_counter()
        duration_ms = round((t1 - t0) * 1000, 2)
        response_times.append(duration_ms)

        v2_data = v2_res.json() if v2_res.status_code == 200 else {}

        # Execute V1 Predict API call
        v1_res = client.post("/predict", json={"title": title, "text": text})
        v1_data = v1_res.json() if v1_res.status_code == 200 else {}

        # Parse V2 outputs
        extracted_claims = v2_data.get("claims", [])
        overall_assessment = v2_data.get("overall_assessment", "UNVERIFIED")
        assessment_summary = v2_data.get("assessment_summary", "")
        has_conflict = v2_data.get("has_conflict", False)
        service_status = v2_data.get("service_status", {})
        article_svm = v2_data.get("linguistic_signal")

        # Parse Evidence Details
        fact_check_count = 0
        live_news_count = 0
        supporting_count = 0
        contradicting_count = 0
        neutral_count = 0
        relevant_count = 0
        irrelevant_count = 0
        retrieved_evidence_summary = []

        overall_uncertainty = "HIGH"
        overall_strength = "NONE"
        strength_rank = {"NONE": 0, "LIMITED": 1, "MODERATE": 2, "STRONG": 3}
        current_max_rank = -1

        for c in extracted_claims:
            claim_verdict = c.get("verdict")
            uncertainty_level = str(c.get("uncertainty_level", "HIGH"))
            evidence_strength = str(c.get("evidence_strength", "NONE"))

            overall_uncertainty = uncertainty_level
            rk = strength_rank.get(evidence_strength.upper(), 0)
            if rk > current_max_rank:
                current_max_rank = rk
                overall_strength = evidence_strength

            ev_list = c.get("evidence", [])
            for ev in ev_list:
                src_type = ev.get("source_type")
                if src_type == "FACT_CHECK_API":
                    fact_check_count += 1
                elif src_type == "LIVE_NEWS_SEARCH":
                    live_news_count += 1

                rel_score = ev.get("relevance_score")
                if rel_score is not None and rel_score > 0.0:
                    relevant_count += 1
                else:
                    # In current pipeline, matched evidence is considered relevant
                    relevant_count += 1

                stance = ev.get("stance")
                if stance == "SUPPORTS":
                    supporting_count += 1
                elif stance == "CONTRADICTS":
                    contradicting_count += 1
                else:
                    neutral_count += 1

                retrieved_evidence_summary.append({
                    "id": ev.get("id"),
                    "publisher": ev.get("publisher"),
                    "domain": ev.get("domain"),
                    "url": sanitize_secret(ev.get("url")),
                    "title": ev.get("title"),
                    "stance": stance,
                    "source_type": src_type
                })

        total_evidence_count = fact_check_count + live_news_count

        # V1 metrics
        v1_prediction = v1_data.get("prediction", "UNKNOWN")
        v1_confidence = v1_data.get("confidence", 0.0)

        # Correctness
        is_correct = (overall_assessment == ground_truth)

        # Failure Classification
        fail_class = classify_failure(
            case=case,
            v2_assessment=overall_assessment,
            ground_truth=ground_truth,
            extracted_claims=extracted_claims,
            fact_check_count=fact_check_count,
            live_news_count=live_news_count,
            total_evidence_count=total_evidence_count,
            relevant_count=relevant_count,
            supporting_count=supporting_count,
            contradicting_count=contradicting_count,
            neutral_count=neutral_count,
            has_conflict=has_conflict,
            service_status=service_status
        )

        entry = {
            "id": case_id,
            "category": category,
            "title": title,
            "text": text,
            "ground_truth": ground_truth,
            "ground_truth_source": ground_truth_source,
            "source_date": source_date,
            # FIX (Stage 30J): store the actual number of extracted claims so that
            # claim_extraction_success_rate is computed from real data, not a default.
            "claim_count": len(extracted_claims),
            "fact_check_attempted": True,
            "fact_check_count": fact_check_count,
            "live_news_attempted": True,
            "live_news_count": live_news_count,
            "service_status": service_status,
            "total_evidence_count": total_evidence_count,
            "relevant_evidence_count": relevant_count,
            "irrelevant_evidence_count": irrelevant_count,
            "supporting_evidence_count": supporting_count,
            "contradicting_evidence_count": contradicting_count,
            "neutral_evidence_count": neutral_count,
            "v2_overall_assessment": overall_assessment,
            "v2_assessment_summary": assessment_summary,
            "uncertainty_level": overall_uncertainty,
            "evidence_strength": overall_strength,
            "has_conflict": has_conflict,
            "response_time_ms": duration_ms,
            "v1_prediction": v1_prediction,
            "v1_confidence": v1_confidence,
            "article_svm_signal": article_svm,
            "retrieved_evidence": retrieved_evidence_summary,
            "is_correct": is_correct,
            "failure_classification": fail_class,
            "notes": case.get("notes", "")
        }

        case_results.append(entry)

    # 3. Calculate Comprehensive Metrics
    total_cases = len(case_results)

    # Claim extraction success — FIX (Stage 30J): use the stored claim_count field
    # (no default fallback; field is now always written to case_results above).
    extraction_success_count = sum(1 for c in case_results if c["claim_count"] > 0)
    extraction_success_rate = round((extraction_success_count / total_cases) * 100, 2)

    # Retrieval hit rates
    fact_check_hits = sum(1 for c in case_results if c["fact_check_count"] > 0)
    fact_check_hit_rate = round((fact_check_hits / total_cases) * 100, 2)

    live_news_hits = sum(1 for c in case_results if c["live_news_count"] > 0)
    live_news_hit_rate = round((live_news_hits / total_cases) * 100, 2)

    evidence_relevant_cases = sum(1 for c in case_results if c["relevant_evidence_count"] > 0)
    evidence_relevance_rate = round((evidence_relevant_cases / total_cases) * 100, 2)

    # Evidence Stance Counts
    total_supports = sum(c["supporting_evidence_count"] for c in case_results)
    total_contradicts = sum(c["contradicting_evidence_count"] for c in case_results)
    total_neutrals = sum(c["neutral_evidence_count"] for c in case_results)
    total_all_evidence = sum(c["total_evidence_count"] for c in case_results)

    # FIX (Stage 30J): renamed to "retrieved_evidence_stance_coverage" to make the
    # denominator explicit — this is the % of RETRIEVED evidence items that have a
    # stance assigned (S+C+N), not a per-case or per-all-64 coverage figure.
    # Denominator = total_all_evidence (retrieved items only; cases with 0 evidence excluded).
    retrieved_evidence_stance_count = total_supports + total_contradicts + total_neutrals
    stance_coverage_rate = round(
        (retrieved_evidence_stance_count / total_all_evidence) * 100, 2
    ) if total_all_evidence > 0 else 0.0

    verdict_coverage_count = sum(1 for c in case_results if c["v2_overall_assessment"] in ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"])
    verdict_coverage_rate = round((verdict_coverage_count / total_cases) * 100, 2)

    # Accuracy 1: Overall accuracy across all cases
    overall_correct_count = sum(1 for c in case_results if c["is_correct"])
    overall_accuracy = round((overall_correct_count / total_cases) * 100, 2)

    # Accuracy 2: Accuracy on cases where ALL external services returned ok.
    # FIX (Stage 30J): when no fully-clean cases exist (e.g. GDELT failed on every
    # case), report "N/A — no fully-clean cases" instead of 0.0, which was
    # misleading (implying 0% accuracy rather than the metric being inapplicable).
    valid_cases = [c for c in case_results if all(v == "ok" for v in c["service_status"].values())]
    if valid_cases:
        valid_correct_count = sum(1 for c in valid_cases if c["is_correct"])
        valid_accuracy: object = round((valid_correct_count / len(valid_cases)) * 100, 2)
    else:
        valid_accuracy = "N/A — no fully-clean cases"

    unverified_count = sum(1 for c in case_results if c["v2_overall_assessment"] == "UNVERIFIED")
    unverified_rate = round((unverified_count / total_cases) * 100, 2)

    # False Positive: CONTRADICTED ground truth incorrectly predicted as SUPPORTED
    false_ground_truth_cases = [c for c in case_results if c["ground_truth"] == "CONTRADICTED"]
    false_positives = sum(1 for c in false_ground_truth_cases if c["v2_overall_assessment"] == "SUPPORTED")
    false_positive_rate = round((false_positives / len(false_ground_truth_cases)) * 100, 2) if false_ground_truth_cases else 0.0

    # False Negative: SUPPORTED ground truth incorrectly predicted as CONTRADICTED
    true_ground_truth_cases = [c for c in case_results if c["ground_truth"] == "SUPPORTED"]
    false_negatives = sum(1 for c in true_ground_truth_cases if c["v2_overall_assessment"] == "CONTRADICTED")
    false_negative_rate = round((false_negatives / len(true_ground_truth_cases)) * 100, 2) if true_ground_truth_cases else 0.0

    conflict_count = sum(1 for c in case_results if c["has_conflict"])
    conflict_rate = round((conflict_count / total_cases) * 100, 2)

    # Measures cases where the LIVE_NEWS (NewsAPI) provider returned a non-ok status.
    live_news_unavailability_count = sum(
        1 for c in case_results
        if c["service_status"].get("live_news_api", "ok") != "ok"
    )
    live_news_unavailability_rate = round((live_news_unavailability_count / total_cases) * 100, 2)
    gdelt_unavailability_count = live_news_unavailability_count
    gdelt_unavailability_rate = live_news_unavailability_rate

    # Precision, Recall, F1
    def calc_p_r_f1(pred_label: str, gt_label: str):
        tp = sum(1 for c in case_results if c["v2_overall_assessment"] == pred_label and c["ground_truth"] == gt_label)
        fp = sum(1 for c in case_results if c["v2_overall_assessment"] == pred_label and c["ground_truth"] != gt_label)
        fn = sum(1 for c in case_results if c["v2_overall_assessment"] != pred_label and c["ground_truth"] == gt_label)

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return round(precision, 2), round(recall, 2), round(f1, 2)

    p_supp, r_supp, f1_supp = calc_p_r_f1("SUPPORTED", "SUPPORTED")
    p_cont, r_cont, f1_cont = calc_p_r_f1("CONTRADICTED", "CONTRADICTED")
    p_unv, r_unv, f1_unv = calc_p_r_f1("UNVERIFIED", "UNVERIFIED")

    # Macro F1
    macro_f1 = round((f1_supp + f1_cont + f1_unv) / 3, 2)

    # Latency
    avg_latency = round(statistics.mean(response_times), 2)
    median_latency = round(statistics.median(response_times), 2)
    p95_latency = round(calculate_p95(response_times), 2)

    # Confusion Matrix
    # Matrix format: Ground Truth (rows) vs Predicted (columns)
    labels = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]
    confusion_matrix = {gt: {pred: 0 for pred in labels} for gt in labels}

    for c in case_results:
        gt = c["ground_truth"]
        pred = c["v2_overall_assessment"]
        if gt in confusion_matrix and pred in confusion_matrix[gt]:
            confusion_matrix[gt][pred] += 1

    # Failure Classification Breakdown
    failure_counts = {
        "claim_extraction_failure": 0,
        "fact_check_retrieval_failure": 0,
        "live_news_retrieval_failure": 0,
        "irrelevant_evidence": 0,
        "stance_error": 0,
        "verdict_logic_error": 0,
        "source_conflict": 0,
        "insufficient_evidence": 0,
        "ground_truth_ambiguity": 0,
        "service_failure": 0,
        "none": 0
    }
    for c in case_results:
        fc = c["failure_classification"]
        failure_counts[fc] = failure_counts.get(fc, 0) + 1

    # Assembly of Metrics
    metrics = {
        "evaluation_mode": eval_mode_title,
        "google_fact_check_api_status": fc_api_status,
        "news_api_status": news_api_status,
        "total_cases": total_cases,
        "claim_extraction_success_rate": extraction_success_rate,
        "fact_check_hit_rate": fact_check_hit_rate,
        "live_news_hit_rate": live_news_hit_rate,
        "evidence_relevance_rate": evidence_relevance_rate,
        # FIX (Stage 30J): renamed from stance_determination_coverage.
        # Denominator = total retrieved evidence items (not total cases).
        "retrieved_evidence_stance_coverage": stance_coverage_rate,
        "supports_evidence_count": total_supports,
        "contradicts_evidence_count": total_contradicts,
        "neutral_evidence_count": total_neutrals,
        "total_all_evidence_count": total_all_evidence,
        "verdict_coverage_rate": verdict_coverage_rate,
        "overall_ground_truth_accuracy": overall_accuracy,
        # FIX (Stage 30J): renamed and changed sentinel value — "N/A" when no
        # fully-clean cases exist, not 0.0 (which implied 0% accuracy).
        "accuracy_fully_clean_cases": valid_accuracy,
        "fully_clean_case_count": len(valid_cases),
        "unverified_rate": unverified_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "conflict_rate": conflict_rate,
        "conflict_count": conflict_count,
        "live_news_unavailability_rate": live_news_unavailability_rate,
        "live_news_unavailability_count": live_news_unavailability_count,
        "gdelt_unavailability_rate": gdelt_unavailability_rate,
        "gdelt_unavailability_count": gdelt_unavailability_count,
        "precision_supported": p_supp,
        "recall_supported": r_supp,
        "f1_supported": f1_supp,
        "precision_contradicted": p_cont,
        "recall_contradicted": r_cont,
        "f1_contradicted": f1_cont,
        "precision_unverified": p_unv,
        "recall_unverified": r_unv,
        "f1_unverified": f1_unv,
        "macro_f1_score": macro_f1,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "confusion_matrix": confusion_matrix,
        "failure_categories": failure_counts
    }

    output_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "evaluation_mode": eval_mode_title,
        "metrics": metrics,
        "case_results": sanitize_secret(case_results)
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Print Summary Report
    print("\n" + "=" * 80)
    print(f"STAGE 32 — EVALUATION SUMMARY REPORT [{eval_mode_title}]")
    print("=" * 80)
    print(f"API Configuration:")
    print(f"  - Google Fact Check API Key: {fc_api_status.upper()}")
    print(f"  - NewsAPI Key:               {news_api_status.upper()}")
    print("-" * 80)
    print(f"Dataset & Metrics Overview:")
    print(f"  Total Benchmark Cases:              {total_cases}")
    print(f"  Claim Extraction Success Rate:      {extraction_success_rate}%")
    print(f"  Fact-Check Retrieval Hit Rate:       {fact_check_hit_rate}%")
    print(f"  Live-News Retrieval Hit Rate:        {live_news_hit_rate}%")
    print(f"  Evidence Relevance Rate:            {evidence_relevance_rate}%")
    print(f"  Retrieved-Evidence Stance Coverage: {stance_coverage_rate}% (of retrieved items)")
    print(f"  SUPPORTS Evidence Count:            {total_supports}")
    print(f"  CONTRADICTS Evidence Count:         {total_contradicts}")
    print(f"  NEUTRAL Evidence Count:             {total_neutrals}")
    print(f"  Verdict Coverage Rate:              {verdict_coverage_rate}%")
    print(f"  Ground-Truth Accuracy (All Cases):  {overall_accuracy}%")
    print(f"  Accuracy (Fully-Clean Cases Only):  {valid_accuracy}")
    print(f"  UNVERIFIED Rate:                    {unverified_rate}%")
    print(f"  False Positive Rate (Fake -> Real): {false_positive_rate}%")
    print(f"  False Negative Rate (Real -> Fake): {false_negative_rate}%")
    print(f"  Conflict Rate:                      {conflict_rate}% ({conflict_count} cases)")
    print(f"  Live-News Unavailability Rate:      {live_news_unavailability_rate}% ({live_news_unavailability_count} cases)")
    print("-" * 80)
    print("Classification Metrics (Precision / Recall / F1):")
    print(f"  - SUPPORTED:    Precision: {p_supp}% | Recall: {r_supp}% | F1: {f1_supp}")
    print(f"  - CONTRADICTED: Precision: {p_cont}% | Recall: {r_cont}% | F1: {f1_cont}")
    print(f"  - UNVERIFIED:   Precision: {p_unv}% | Recall: {r_unv}% | F1: {f1_unv}")
    print(f"  - MACRO F1:     {macro_f1}")
    print("-" * 80)
    print("Latency Metrics:")
    print(f"  - Average Latency:  {avg_latency} ms")
    print(f"  - Median Latency:   {median_latency} ms")
    print(f"  - p95 Latency:      {p95_latency} ms")
    print("-" * 80)
    print("Confusion Matrix (Ground Truth [rows] vs System Predicted [cols]):")
    print("                 Pred SUPPORTED  Pred CONTRADICTED  Pred UNVERIFIED")
    for gt_lbl in labels:
        row = confusion_matrix[gt_lbl]
        print(f"  GT {gt_lbl:<12}  {row['SUPPORTED']:<14}  {row['CONTRADICTED']:<17}  {row['UNVERIFIED']}")
    print("-" * 80)
    print("Failure Classification Breakdown:")
    for cat_name, cat_count in failure_counts.items():
        print(f"  - {cat_name:<30}: {cat_count}")
    print("=" * 80)
    print(f"Results successfully saved to: {results_path}")

    return output_payload


if __name__ == "__main__":
    is_mock = "--mock" in sys.argv
    limit_val = None
    start_val = 0
    out_file = None

    if "--limit" in sys.argv:
        try:
            l_idx = sys.argv.index("--limit")
            limit_val = int(sys.argv[l_idx + 1])
        except (IndexError, ValueError):
            pass

    if "--start" in sys.argv:
        try:
            s_idx = sys.argv.index("--start")
            start_val = int(sys.argv[s_idx + 1])
        except (IndexError, ValueError):
            pass

    if "--output" in sys.argv:
        try:
            o_idx = sys.argv.index("--output")
            out_file = sys.argv[o_idx + 1]
        except IndexError:
            pass

    run_real_world_evaluation(
        mock_mode=is_mock,
        start_idx=start_val,
        limit=limit_val,
        output_file=out_file
    )
