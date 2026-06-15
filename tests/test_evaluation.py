"""
Tests for the evaluation metrics module.

Covers: Precision@K, Recall@K, MRR, faithfulness score,
and the retrieval evaluation runner.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    EVAL_DATASET,
    faithfulness_score,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ─── Precision@K ─────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["pricing_overview", "zones", "booking_process", "location"]
        relevant = ["pricing_overview", "zones", "booking_process", "location"]
        assert precision_at_k(retrieved, relevant, k=4) == 1.0

    def test_none_relevant(self):
        retrieved = ["faq", "amenities", "contact", "policies"]
        relevant = ["pricing_overview"]
        assert precision_at_k(retrieved, relevant, k=4) == 0.0

    def test_half_relevant(self):
        retrieved = ["pricing_overview", "faq", "zones", "contact"]
        relevant = ["pricing_overview", "zones"]
        p = precision_at_k(retrieved, relevant, k=4)
        assert abs(p - 0.5) < 1e-6

    def test_k_equals_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_top_1_relevant(self):
        retrieved = ["pricing_overview", "faq", "zones", "contact"]
        relevant = ["pricing_overview"]
        assert precision_at_k(retrieved, relevant, k=1) == 1.0

    def test_top_1_irrelevant(self):
        retrieved = ["faq", "pricing_overview", "zones"]
        relevant = ["pricing_overview"]
        assert precision_at_k(retrieved, relevant, k=1) == 0.0


# ─── Recall@K ────────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        retrieved = ["pricing_overview", "zones", "booking_process"]
        relevant = ["pricing_overview", "zones"]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_retrieved(self):
        retrieved = ["faq", "amenities"]
        relevant = ["pricing_overview", "zones"]
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_recall(self):
        retrieved = ["pricing_overview", "faq", "contact"]
        relevant = ["pricing_overview", "zones"]
        r = recall_at_k(retrieved, relevant, k=3)
        assert abs(r - 0.5) < 1e-6

    def test_empty_relevant_set(self):
        # If there are no relevant docs, recall is 1 by convention
        assert recall_at_k(["anything"], [], k=4) == 1.0

    def test_k_limits_consideration(self):
        # With k=1 we only look at the first retrieved doc
        retrieved = ["faq", "pricing_overview", "zones"]
        relevant = ["pricing_overview"]
        assert recall_at_k(retrieved, relevant, k=1) == 0.0
        assert recall_at_k(retrieved, relevant, k=2) == 1.0


# ─── MRR ─────────────────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["pricing_overview", "faq"], ["pricing_overview"]) == 1.0

    def test_second_position(self):
        rr = reciprocal_rank(["faq", "pricing_overview", "zones"], ["pricing_overview"])
        assert abs(rr - 0.5) < 1e-6

    def test_third_position(self):
        rr = reciprocal_rank(["faq", "zones", "pricing_overview"], ["pricing_overview"])
        assert abs(rr - 1 / 3) < 1e-6

    def test_not_found(self):
        assert reciprocal_rank(["faq", "zones"], ["pricing_overview"]) == 0.0

    def test_multiple_relevant_uses_first(self):
        retrieved = ["faq", "pricing_overview", "zones"]
        relevant = ["zones", "pricing_overview"]  # pricing_overview is at rank 2, zones at rank 3
        rr = reciprocal_rank(retrieved, relevant)
        # First hit is pricing_overview at rank 2
        assert abs(rr - 0.5) < 1e-6


# ─── Faithfulness ────────────────────────────────────────────────────────────

class TestFaithfulnessScore:
    def test_identical_text_is_faithful(self):
        text = "Parking costs five dollars per hour in Zone B."
        score = faithfulness_score(text, text)
        assert score == 1.0

    def test_completely_different_text_low_score(self):
        answer = "elephants swim in the ocean at midnight"
        context = "Parking is available at 123 Innovation Boulevard."
        score = faithfulness_score(answer, context)
        assert score < 0.2

    def test_empty_answer_returns_zero(self):
        assert faithfulness_score("", "some context") == 0.0

    def test_empty_context_returns_zero(self):
        assert faithfulness_score("some answer", "") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        answer = "The parking facility is located on Innovation Boulevard near City Hall."
        context = "SmartPark City Center is located at 123 Innovation Boulevard, City Center, opposite City Hall."
        score = faithfulness_score(answer, context)
        assert 0.0 < score < 1.0


# ─── Eval dataset ─────────────────────────────────────────────────────────────

class TestEvalDataset:
    def test_dataset_not_empty(self):
        assert len(EVAL_DATASET) >= 5

    def test_each_entry_has_required_keys(self):
        for entry in EVAL_DATASET:
            assert "query" in entry
            assert "relevant_sections" in entry
            assert isinstance(entry["relevant_sections"], list)
            assert len(entry["relevant_sections"]) >= 1


# ─── Retrieval evaluation (integration) ──────────────────────────────────────

class TestRetrievalEvaluation:
    """Run the retrieval evaluator against the real vector store."""

    @pytest.fixture(autouse=True)
    def _patch_store_path(self, tmp_path, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setattr(cfg, "COLLECTION_NAME", "eval_test")
        import src.rag.vectorstore as vs_module
        vs_module._store_instance = None

    def test_evaluate_retrieval_runs(self):
        from src.evaluation.metrics import evaluate_retrieval
        report = evaluate_retrieval(k=4)
        assert report.total_queries == len(EVAL_DATASET)

    def test_report_metrics_in_range(self):
        from src.evaluation.metrics import evaluate_retrieval
        report = evaluate_retrieval(k=4)
        for attr in ("mean_precision_at_k", "mean_recall_at_k", "mrr"):
            val = getattr(report, attr)
            assert 0.0 <= val <= 1.0, f"{attr} out of [0, 1] range: {val}"

    def test_report_latency_positive(self):
        from src.evaluation.metrics import evaluate_retrieval
        report = evaluate_retrieval(k=4)
        assert report.mean_retrieval_latency_ms > 0

    def test_report_has_per_query_breakdown(self):
        from src.evaluation.metrics import evaluate_retrieval
        report = evaluate_retrieval(k=4)
        assert len(report.per_query) == len(EVAL_DATASET)

    def test_report_to_dict_serialisable(self):
        import json

        from src.evaluation.metrics import evaluate_retrieval
        report = evaluate_retrieval(k=2)
        d = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 10
