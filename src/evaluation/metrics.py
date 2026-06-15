"""
RAG Evaluation Module — Stage 5

Implements:
  • Precision@K  — fraction of retrieved docs that are relevant
  • Recall@K     — fraction of known-relevant docs that appear in top-K
  • MRR          — Mean Reciprocal Rank
  • Latency      — end-to-end response time
  • Faithfulness — simple lexical overlap between answer and context

A test dataset with ground-truth labels is included so evaluation
can run without any manual annotation.
"""
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ─── Evaluation dataset ───────────────────────────────────────────────────────

# Each entry: query + list of section keys that are ground-truth relevant
EVAL_DATASET = [
    {
        "query": "What are the parking prices?",
        "relevant_sections": ["pricing_overview", "zones"],
        "reference_answer": "Zone A premium costs $6 per hour with a $35 daily maximum.",
    },
    {
        "query": "What are the working hours?",
        "relevant_sections": ["working_hours"],
        "reference_answer": "The facility is open 24/7. Staffed booth: Mon–Fri 07:00–22:00.",
    },
    {
        "query": "Where is the parking located?",
        "relevant_sections": ["location"],
        "reference_answer": "123 Innovation Boulevard, City Center, opposite City Hall.",
    },
    {
        "query": "Are there EV charging stations?",
        "relevant_sections": ["zones", "amenities"],
        "reference_answer": "Yes, Zone D on the rooftop has 40 Level 2 AC and 10 DC fast chargers.",
    },
    {
        "query": "What is the cancellation policy?",
        "relevant_sections": ["booking_process"],
        "reference_answer": "Cancel 24+ hours before for a full refund; within 24 hours incurs a 50% fee.",
    },
    {
        "query": "Is there disabled parking?",
        "relevant_sections": ["zones"],
        "reference_answer": "Zone E provides 20 extra-wide accessible bays; free for valid disability badge holders.",
    },
    {
        "query": "How do I book a parking space?",
        "relevant_sections": ["booking_process"],
        "reference_answer": "Reservations can be made via the chatbot, website, or at the staffed booth.",
    },
    {
        "query": "What security measures are in place?",
        "relevant_sections": ["amenities"],
        "reference_answer": "24/7 CCTV, trained security staff, SOS buttons every 20 metres, and anti-tailgating barriers.",
    },
    {
        "query": "What is the maximum parking duration?",
        "relevant_sections": ["booking_process", "policies"],
        "reference_answer": "The maximum single reservation period is 30 days.",
    },
    {
        "query": "How do I get to SmartPark from the train station?",
        "relevant_sections": ["location"],
        "reference_answer": "Central Station is 200 metres west. Take Exit B for a 3-minute walk.",
    },
]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class QueryMetrics:
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    retrieved_sections: List[str]
    relevant_sections: List[str]
    retrieval_latency_ms: float
    generation_latency_ms: float = 0.0
    faithfulness_score: float = 0.0
    answer: str = ""


@dataclass
class EvaluationReport:
    mean_precision_at_k: float
    mean_recall_at_k: float
    mrr: float
    mean_retrieval_latency_ms: float
    mean_generation_latency_ms: float
    mean_faithfulness: float
    per_query: List[QueryMetrics] = field(default_factory=list)
    k: int = 4
    total_queries: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "   SmartPark RAG Evaluation Report",
            "=" * 60,
            f"  Queries evaluated : {self.total_queries}",
            f"  K (top-K)         : {self.k}",
            f"  Precision@{self.k}       : {self.mean_precision_at_k:.3f}",
            f"  Recall@{self.k}          : {self.mean_recall_at_k:.3f}",
            f"  MRR               : {self.mrr:.3f}",
            f"  Faithfulness      : {self.mean_faithfulness:.3f}",
            f"  Avg retrieval lat : {self.mean_retrieval_latency_ms:.1f} ms",
            f"  Avg generation lat: {self.mean_generation_latency_ms:.1f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "total_queries": self.total_queries,
            "mean_precision_at_k": round(self.mean_precision_at_k, 4),
            "mean_recall_at_k": round(self.mean_recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_retrieval_latency_ms": round(self.mean_retrieval_latency_ms, 2),
            "mean_generation_latency_ms": round(self.mean_generation_latency_ms, 2),
            "per_query": [
                {
                    "query": q.query,
                    "precision_at_k": round(q.precision_at_k, 4),
                    "recall_at_k": round(q.recall_at_k, 4),
                    "reciprocal_rank": round(q.reciprocal_rank, 4),
                    "retrieval_latency_ms": round(q.retrieval_latency_ms, 2),
                }
                for q in self.per_query
            ],
        }


# ─── Metric functions ─────────────────────────────────────────────────────────

def precision_at_k(retrieved_sections: List[str], relevant_sections: List[str], k: int) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:K]| / K
    """
    if k == 0:
        return 0.0
    top_k = retrieved_sections[:k]
    hits = sum(1 for sec in top_k if sec in relevant_sections)
    return hits / k


def recall_at_k(retrieved_sections: List[str], relevant_sections: List[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
    """
    if not relevant_sections:
        return 1.0
    top_k = retrieved_sections[:k]
    hits = sum(1 for sec in relevant_sections if sec in top_k)
    return hits / len(relevant_sections)


def reciprocal_rank(retrieved_sections: List[str], relevant_sections: List[str]) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_document (0 if none found).
    """
    for rank, sec in enumerate(retrieved_sections, start=1):
        if sec in relevant_sections:
            return 1.0 / rank
    return 0.0


def faithfulness_score(answer: str, context: str) -> float:
    """
    Simple lexical faithfulness: fraction of unique content words in the
    answer that also appear in the context.

    This is a lightweight proxy — in production, replace with an LLM-based
    NLI (Natural Language Inference) faithfulness checker.
    """
    if not answer or not context:
        return 0.0

    # Tokenise to lowercase words, removing punctuation
    import re
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "as", "from", "into", "that", "this", "it", "its", "and", "or", "but",
        "not", "no", "so", "if", "then", "than", "there", "their", "they",
        "he", "she", "we", "you", "i", "my", "your", "our", "his", "her",
    }

    def tokenise(text: str):
        return {
            w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)
            if w.lower() not in stop_words
        }

    answer_words = tokenise(answer)
    context_words = tokenise(context)
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


# ─── Retrieval-only evaluator ─────────────────────────────────────────────────

def evaluate_retrieval(k: int = 4) -> EvaluationReport:
    """
    Run retrieval-only evaluation (no LLM call) across the full eval dataset.
    Measures Precision@K, Recall@K, MRR, and retrieval latency.
    """
    from src.rag.retriever import retrieve

    per_query_metrics: List[QueryMetrics] = []

    for item in EVAL_DATASET:
        query = item["query"]
        relevant = item["relevant_sections"]

        t0 = time.perf_counter()
        result = retrieve(query, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_sections = [doc.metadata.get("section", "") for doc in result.documents]

        p_at_k = precision_at_k(retrieved_sections, relevant, k)
        r_at_k = recall_at_k(retrieved_sections, relevant, k)
        rr = reciprocal_rank(retrieved_sections, relevant)

        per_query_metrics.append(QueryMetrics(
            query=query,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            reciprocal_rank=rr,
            retrieved_sections=retrieved_sections,
            relevant_sections=relevant,
            retrieval_latency_ms=latency_ms,
        ))
        logger.debug(
            "Query: %r | P@%d=%.3f | R@%d=%.3f | RR=%.3f | %.1f ms",
            query, k, p_at_k, k, r_at_k, rr, latency_ms,
        )

    latencies = [m.retrieval_latency_ms for m in per_query_metrics]
    report = EvaluationReport(
        mean_precision_at_k=statistics.mean(m.precision_at_k for m in per_query_metrics),
        mean_recall_at_k=statistics.mean(m.recall_at_k for m in per_query_metrics),
        mrr=statistics.mean(m.reciprocal_rank for m in per_query_metrics),
        mean_retrieval_latency_ms=statistics.mean(latencies),
        mean_generation_latency_ms=0.0,
        mean_faithfulness=0.0,
        per_query=per_query_metrics,
        k=k,
        total_queries=len(EVAL_DATASET),
    )
    return report


def evaluate_full_pipeline(k: int = 4) -> EvaluationReport:
    """
    Full pipeline evaluation including LLM generation and faithfulness scoring.
    Uses Groq or MockLLM in offline mode.
    """
    from langchain_core.output_parsers import StrOutputParser

    import src.config as cfg
    from src.chatbot.llm import get_llm
    from src.chatbot.prompts import RAG_ANSWER_PROMPT
    from src.database.models import init_db
    from src.database.operations import format_pricing_context
    from src.rag.retriever import retrieve

    init_db(cfg.SQLITE_DB_PATH)
    llm = get_llm()
    chain = RAG_ANSWER_PROMPT | llm | StrOutputParser()

    per_query_metrics: List[QueryMetrics] = []

    for item in EVAL_DATASET:
        query = item["query"]
        relevant = item["relevant_sections"]

        # Retrieval
        t0 = time.perf_counter()
        result = retrieve(query, k=k)
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_sections = [doc.metadata.get("section", "") for doc in result.documents]

        # Generation
        dynamic_ctx = format_pricing_context(cfg.SQLITE_DB_PATH)
        t1 = time.perf_counter()
        try:
            answer = chain.invoke({
                "user_message": query,
                "static_context": result.formatted_context,
                "dynamic_context": dynamic_ctx,
            })
        except Exception as exc:
            logger.warning("Generation failed for query %r: %s", query, exc)
            answer = ""
        generation_latency_ms = (time.perf_counter() - t1) * 1000

        p_at_k = precision_at_k(retrieved_sections, relevant, k)
        r_at_k = recall_at_k(retrieved_sections, relevant, k)
        rr = reciprocal_rank(retrieved_sections, relevant)
        faith = faithfulness_score(answer, result.formatted_context)

        per_query_metrics.append(QueryMetrics(
            query=query,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            reciprocal_rank=rr,
            retrieved_sections=retrieved_sections,
            relevant_sections=relevant,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            faithfulness_score=faith,
            answer=answer,
        ))

    def _safe_mean(values):
        return statistics.mean(values) if values else 0.0

    report = EvaluationReport(
        mean_precision_at_k=_safe_mean([m.precision_at_k for m in per_query_metrics]),
        mean_recall_at_k=_safe_mean([m.recall_at_k for m in per_query_metrics]),
        mrr=_safe_mean([m.reciprocal_rank for m in per_query_metrics]),
        mean_retrieval_latency_ms=_safe_mean([m.retrieval_latency_ms for m in per_query_metrics]),
        mean_generation_latency_ms=_safe_mean([m.generation_latency_ms for m in per_query_metrics]),
        mean_faithfulness=_safe_mean([m.faithfulness_score for m in per_query_metrics]),
        per_query=per_query_metrics,
        k=k,
        total_queries=len(EVAL_DATASET),
    )
    return report


def save_report(report: EvaluationReport, output_path: str = "evaluation_report.json") -> None:
    """Persist the evaluation report as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Evaluation report saved to %s", output_path)
