"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas

    Args:
        rag_pipeline: Hàm callable dạng `generate_with_citation(question) -> dict`
            (import từ `src.task10_generation`), trả về {'answer': str, 'sources': list[dict]}.
        golden_dataset: List các {'question', 'expected_answer', 'expected_context'}.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = rag_pipeline(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
    - hybrid_rerank: hybrid search (semantic + lexical) + reranking
    - dense_only: chỉ dense/semantic search, không reranking

    Args:
        rag_pipeline: `generate_with_citation` — phải nhận keyword arg `use_reranking`
            (xem `src/task10_generation.py`).
        golden_dataset: Golden dataset.

    Returns:
        dict {config_name: pandas.DataFrame} — mỗi DataFrame là kết quả RAGAS
        (1 dòng/câu hỏi) của config đó, dùng cho `export_results`.
    """
    from functools import partial

    configs = {
        "hybrid_rerank": {"use_reranking": True},
        "dense_only": {"use_reranking": False},
    }

    results = {}
    for config_name, params in configs.items():
        print(f"  → Evaluating config: {config_name} ({params})")
        pipeline_variant = partial(rag_pipeline, **params)
        results[config_name] = evaluate_with_ragas(pipeline_variant, golden_dataset)

    return results


# =============================================================================
# Export Results
# =============================================================================

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def export_results(results, comparison: dict):
    """
    Export evaluation results to results.md

    Args:
        results: pandas.DataFrame trả về bởi `evaluate_with_ragas` (1 dòng/câu hỏi).
        comparison: dict {config_name: pandas.DataFrame} trả về bởi `compare_configs`.
    """
    lines = ["# RAG Evaluation Results", ""]
    lines.append(f"Golden dataset: {len(results)} câu hỏi.")
    lines.append("")

    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for col in METRIC_COLUMNS:
        if col in results.columns:
            lines.append(f"| {col} | {results[col].mean():.3f} |")
    lines.append("")

    lines.append("## A/B Comparison")
    lines.append("")
    lines.append("| Config | " + " | ".join(METRIC_COLUMNS) + " |")
    lines.append("|--------|" + "|".join(["-------"] * len(METRIC_COLUMNS)) + "|")
    for config_name, df in comparison.items():
        row = [
            f"{df[col].mean():.3f}" if col in df.columns else "N/A"
            for col in METRIC_COLUMNS
        ]
        lines.append(f"| {config_name} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Worst Performers")
    lines.append("")
    if "faithfulness" in results.columns:
        worst = results.nsmallest(3, "faithfulness")
        lines.append("| Question | Faithfulness | Answer Relevancy |")
        lines.append("|----------|--------------|-------------------|")
        for _, row in worst.iterrows():
            question = str(row.get("question", "")).replace("|", "/")[:100]
            faithfulness_score = row.get("faithfulness", float("nan"))
            relevancy_score = row.get("answer_relevancy", float("nan"))
            lines.append(f"| {question} | {faithfulness_score:.3f} | {relevancy_score:.3f} |")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "- Xem lại các câu hỏi trong bảng \"Worst Performers\" — điểm faithfulness thấp "
        "thường do context thiếu evidence hoặc retrieval trả nhầm chunk."
    )
    lines.append(
        "- So sánh 2 dòng trong bảng A/B: nếu `dense_only` gần bằng `hybrid_rerank`, "
        "cân nhắc bỏ bước reranking để giảm latency; nếu chênh lệch lớn, giữ reranking."
    )
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    from src.task10_generation import generate_with_citation

    print("Running RAGAS evaluation (overall)...")
    results = evaluate_with_ragas(generate_with_citation, golden_dataset)

    print("Running A/B comparison (hybrid_rerank vs dense_only)...")
    comparison = compare_configs(generate_with_citation, golden_dataset)

    export_results(results, comparison)
    print(f"✓ Results exported to {RESULTS_PATH}")
