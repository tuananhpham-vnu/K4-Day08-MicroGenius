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
import os
import sys
import types
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
EVAL_LOG_PATH = Path(__file__).parent / "eval.json"
RESULTS_VERSION_PREFIX = "results_v"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


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

def evaluate_with_ragas(
    rag_pipeline,
    golden_dataset: list[dict],
    run_name: str = "overall",
) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas

    Args:
        rag_pipeline: Hàm callable dạng `generate_with_citation(question) -> dict`
            (import từ `src.task10_generation`), trả về {'answer': str, 'sources': list[dict]}.
        golden_dataset: List các {'question', 'expected_answer', 'expected_context'}.
    """
    _install_ragas_vertexai_compat()

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    llm = _build_ragas_llm()
    embeddings = LocalHashEmbeddings()
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    eval_records: list[dict] = []
    run_id = _new_run_id(run_name)

    for index, item in enumerate(golden_dataset, 1):
        result = rag_pipeline(item["question"])
        contexts = [c["content"] for c in result["sources"]]
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])
        eval_records.append(
            {
                "index": index,
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "answer": result["answer"],
                "contexts": contexts,
                "source_count": len(result["sources"]),
            }
        )
        _write_eval_log(run_id, run_name, "generating", eval_records)

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    result_df = result.to_pandas()
    if "question" not in result_df.columns:
        result_df.insert(0, "question", eval_data["question"])
    for index, record in enumerate(eval_records):
        if index >= len(result_df):
            continue
        for metric in METRIC_COLUMNS:
            if metric in result_df.columns:
                record[metric] = _json_safe_float(result_df.iloc[index][metric])
    _write_eval_log(run_id, run_name, "scored", eval_records)
    return result_df


def _build_ragas_llm():
    """Build the OpenRouter chat model used by RAGAS evaluation."""
    from langchain_openai import ChatOpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model = os.getenv("RAGAS_LLM_MODEL", "openai/gpt-4o-mini").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if openrouter_key:
        return ChatOpenAI(
            model=openrouter_model,
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_retries=2,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Lab8 RAG Evaluation"),
            },
        )

    raise RuntimeError(
        "RAGAS is configured to use OpenRouter. Set OPENROUTER_API_KEY in .env. "
        f"Existing keys detected: OPENAI_API_KEY={bool(openai_key)}, "
        f"GEMINI_API_KEY={bool(gemini_key)}, "
        f"DEEPSEEK_API_KEY={bool(deepseek_key)}."
    )


class LocalHashEmbeddings:
    """LangChain-compatible local embeddings for RAGAS answer relevancy.

    This is not a mock: it uses the same deterministic hashing embedding as
    Task 5, so RAGAS can compute embedding-based metrics without requiring a
    paid OpenAI embeddings endpoint.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from src.task5_semantic_search import text_embedding

        return [text_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        from src.task5_semantic_search import text_embedding

        return text_embedding(text)


def _install_ragas_vertexai_compat() -> None:
    """Patch a RAGAS 0.1.x import path removed from newer langchain-community.

    ``ragas==0.1.21`` imports ``langchain_community.chat_models.vertexai`` at
    module import time. Newer ``langchain-community`` releases removed that
    module. RAGAS does not need Vertex AI for this project, so this compatibility
    module lets RAGAS import normally while still using its real evaluator.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - only used to satisfy old imports.
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ChatVertexAI is not installed. This project uses RAGAS with "
                "the configured default LLM, not Vertex AI."
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


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
        results[config_name] = evaluate_with_ragas(
            pipeline_variant,
            golden_dataset,
            run_name=config_name,
        )

    return results


# =============================================================================
# Export Results
# =============================================================================

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def export_results(results, comparison: dict, chunking_history: list[dict] | None = None):
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

    if chunking_history:
        lines.append("## Chunking Method Comparison")
        lines.append("")
        lines.append("| Version | Chunking | " + " | ".join(METRIC_COLUMNS) + " |")
        lines.append("|---------|----------|" + "|".join(["-------"] * len(METRIC_COLUMNS)) + "|")
        for item in chunking_history[-5:]:
            scores = item.get("overall_scores", {})
            row = [
                f"{scores[col]:.3f}" if isinstance(scores.get(col), (int, float)) else "N/A"
                for col in METRIC_COLUMNS
            ]
            lines.append(
                f"| {item.get('version', 'N/A')} | {item.get('chunking_method', 'unknown')} | "
                + " | ".join(row)
                + " |"
            )
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


def _new_run_id(run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_name)
    return f"{timestamp}_{safe_name}"


def _write_eval_log(run_id: str, run_name: str, status: str, records: list[dict]) -> None:
    """Persist generation/evaluation progress to eval.json after each question."""
    if EVAL_LOG_PATH.exists():
        try:
            log_data = json.loads(EVAL_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log_data = {"runs": []}
    else:
        log_data = {"runs": []}

    runs = log_data.setdefault("runs", [])
    run_entry = next((run for run in runs if run.get("run_id") == run_id), None)
    if run_entry is None:
        run_entry = {
            "run_id": run_id,
            "run_name": run_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "chunking": _chunking_metadata(),
            "records": [],
        }
        runs.append(run_entry)

    run_entry["status"] = status
    run_entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    run_entry["records"] = records
    EVAL_LOG_PATH.write_text(
        json.dumps(log_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _chunking_metadata() -> dict:
    try:
        from src import task4_chunking_indexing as task4

        return {
            "method": task4.CHUNKING_METHOD,
            "baseline_method": getattr(task4, "BASELINE_CHUNKING_METHOD", "token_text"),
            "chunk_size": task4.CHUNK_SIZE,
            "chunk_overlap": task4.CHUNK_OVERLAP,
        }
    except Exception:
        return {"method": "unknown", "baseline_method": "token_text"}


def _json_safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _mean_scores(df) -> dict:
    scores = {}
    for metric in METRIC_COLUMNS:
        if metric in df.columns:
            scores[metric] = _json_safe_float(df[metric].mean())
    return scores


def _records_from_dataframe(df) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        record = {}
        for key, value in row.to_dict().items():
            safe_float = _json_safe_float(value)
            record[key] = safe_float if safe_float is not None and key in METRIC_COLUMNS else value
        records.append(record)
    return records


def _next_results_version_path() -> Path:
    existing_versions = []
    for path in Path(__file__).parent.glob(f"{RESULTS_VERSION_PREFIX}*.json"):
        version_text = path.stem.replace(RESULTS_VERSION_PREFIX, "", 1)
        if version_text.isdigit():
            existing_versions.append(int(version_text))
    next_version = max(existing_versions, default=0) + 1
    return Path(__file__).parent / f"{RESULTS_VERSION_PREFIX}{next_version}.json"


def export_versioned_results(results, comparison: dict) -> Path:
    """Write results_vN.json for run-to-run comparison."""
    version_path = _next_results_version_path()
    version = version_path.stem.replace(RESULTS_VERSION_PREFIX, "v", 1)
    chunking = _chunking_metadata()
    snapshot = {
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "golden_dataset_size": len(results),
        "chunking_method": chunking.get("method", "unknown"),
        "baseline_chunking_method": chunking.get("baseline_method", "token_text"),
        "chunking": chunking,
        "overall_scores": _mean_scores(results),
        "comparison_scores": {
            config_name: _mean_scores(df) for config_name, df in comparison.items()
        },
        "overall_records": _records_from_dataframe(results),
    }
    version_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return version_path


def load_chunking_history() -> list[dict]:
    """Load versioned JSON snapshots for the Markdown comparison table."""
    history = []
    for path in sorted(Path(__file__).parent.glob(f"{RESULTS_VERSION_PREFIX}*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        item.setdefault("version", path.stem.replace(RESULTS_VERSION_PREFIX, "v", 1))
        history.append(item)
    return history


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    from src.task10_generation import generate_with_citation

    print("Running RAGAS evaluation (overall)...")
    results = evaluate_with_ragas(generate_with_citation, golden_dataset, run_name="overall")

    print("Running A/B comparison (hybrid_rerank vs dense_only)...")
    comparison = compare_configs(generate_with_citation, golden_dataset)

    version_path = export_versioned_results(results, comparison)
    export_results(results, comparison, chunking_history=load_chunking_history())
    print(f"Versioned JSON exported to {version_path}")
    print(f"✓ Results exported to {RESULTS_PATH}")
