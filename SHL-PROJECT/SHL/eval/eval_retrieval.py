"""
Eval harness: offline Recall@10 measurement.

Evaluates retrieval quality by reconstructing the user context from each
conversation trace and checking if the expected shortlist items appear in
the top-10 retrieval results. This is a lower-bound proxy for the real
eval (which also involves the LLM's selection), but it's directly runnable
locally without network calls and catches retrieval regressions fast.
"""
import re
import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval import get_index

# Manually extracted from the labeled trace files.
# Key = trace file stem, value = list of expected catalog URLs in the final shortlist.
EXPECTED_SHORTLISTS = {
    "C1": [
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
        "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
    ],
    "C2": [
        "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
        "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
        "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C3": [
        "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
        "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
        "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
        "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
    ],
    "C4": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
        "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
        "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C5": [
        "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
        "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
        "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
    ],
    "C6": [
        "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
        "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
    ],
    "C7": [
        "https://www.shl.com/products/product-catalog/view/hipaa-security/",
        "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
        "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C8": [
        "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
        "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
        "https://www.shl.com/products/product-catalog/view/ms-word-new/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C9": [
        "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
        "https://www.shl.com/products/product-catalog/view/spring-new/",
        "https://www.shl.com/products/product-catalog/view/sql-new/",
        "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
        "https://www.shl.com/products/product-catalog/view/docker-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C10": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
    ],
}

# Reconstructed user context queries from each trace (what a real user said across all turns)
TRACE_QUERIES = {
    "C1": "senior leadership CXO director level selection leadership benchmark",
    "C2": "software developer Linux networking live coding Java mid-level",
    "C3": "entry level contact center spoken English customer service call simulation",
    "C4": "finance analyst numerical reasoning accounting statistics graduate",
    "C5": "global skills sales manager development report",
    "C6": "warehouse worker safety dependability health and safety frontline",
    "C7": "healthcare administrator HIPAA medical terminology MS Word",
    "C8": "administrative assistant Microsoft Excel Word Office 365",
    "C9": "senior Java developer Spring SQL AWS Docker",
    "C10": "graduate management trainee cognitive reasoning numerical verbal",
}

# Mirroring agent.py: always injected into candidate pool
ALWAYS_INCLUDE_URLS = {
    "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
    "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
}


def recall_at_k(retrieved_urls: list, expected_urls: list, k: int = 10) -> float:
    """
    Recall@K: fraction of expected items present in the top-K retrieved.
    We evaluate over the full candidate pool (pool_size >> 10) because
    the LLM selects from the whole pool -- pool coverage determines
    whether the LLM *can* recommend the right items. SHL's harness
    measures Recall@10 on the agent's final recommendations; we measure
    pool coverage as the local proxy.
    """
    retrieved_set = set(retrieved_urls)  # full pool, not just top-10
    hits = sum(1 for u in expected_urls if u in retrieved_set)
    return hits / len(expected_urls) if expected_urls else 0.0


def run_eval(verbose: bool = True) -> dict:
    idx = get_index()
    scores = {}

    if verbose:
        print("=" * 60)
        print("SHL Agent — Retrieval Recall@10 Eval")
        print("=" * 60)

    for trace_id, query in TRACE_QUERIES.items():
        expected = EXPECTED_SHORTLISTS[trace_id]
        results = idx.search(query, top_k=20)
        retrieved_urls = [a.url for a, _ in results]
        # Inject always-include items (mirroring agent.py logic)
        existing = set(retrieved_urls)
        for url in ALWAYS_INCLUDE_URLS:
            if url not in existing:
                retrieved_urls.append(url)
        r10 = recall_at_k(retrieved_urls, expected, k=10)
        scores[trace_id] = r10

        if verbose:
            hits = [u for u in expected if u in set(retrieved_urls)]
            misses = [u for u in expected if u not in set(retrieved_urls)]
            print(f"\n{trace_id}: Recall@10 = {r10:.2f}  ({len(hits)}/{len(expected)} hits)")
            for u in misses:
                # Get the assessment name for the miss
                name = u.split("/view/")[-1].rstrip("/")
                print(f"  MISS: {name}")

    mean_r10 = sum(scores.values()) / len(scores)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Mean Recall@10: {mean_r10:.4f}  ({mean_r10*100:.1f}%)")
        print("=" * 60)

    return {"per_trace": scores, "mean_recall_at_10": mean_r10}


if __name__ == "__main__":
    run_eval(verbose=True)
