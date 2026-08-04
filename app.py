"""Streamlit demo for the E-commerce Support RAG pipeline.

Flow:
    Streamlit UI -> Task 9 retrieval -> Task 10 citation-ready answer

The demo uses the deterministic local retrieval/generation fallback, so it can
run without an external API key.
"""

from __future__ import annotations

import html
import re
import sys
import unicodedata
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="E-commerce Support RAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .source-card { border: 1px solid #dfe5ec; border-radius: 12px; padding: 12px 14px;
                   margin: 8px 0; background: #fbfcfe; }
    .source-title { font-weight: 700; color: #17324d; margin-bottom: 7px; }
    .source-meta { color: #637083; font-size: 0.82rem; margin-bottom: 8px; }
    .evidence { white-space: pre-wrap; line-height: 1.55; color: #263645; }
    mark { background: #fff0a8; color: #1f2933; padding: 1px 3px; border-radius: 3px; }
    .score-pill { display: inline-block; border-radius: 999px; padding: 2px 8px;
                  background: #e8f3ff; color: #145da0; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


SUGGESTIONS = [
    "Shopee hỗ trợ những phương thức thanh toán nào?",
    "Người mua có thể yêu cầu trả hàng hoặc hoàn tiền trong bao lâu?",
    "Người bán không được đăng bán những sản phẩm nào?",
    "TikTok Shop là gì và ai được phép sử dụng?",
    "Hộ kinh doanh có doanh thu bao nhiêu thì phải nộp thuế?",
]


@st.cache_data(ttl=15)
def get_catalog_stats() -> dict[str, int]:
    """Return lightweight data statistics for the sidebar health panel."""
    landing = PROJECT_ROOT / "data" / "landing"
    standardized = PROJECT_ROOT / "data" / "standardized"
    legal_dir = landing / "legal"
    news_dir = landing / "news"
    return {
        "legal": sum(
            path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".doc"}
            for path in legal_dir.glob("*")
        )
        if legal_dir.exists()
        else 0,
        "news": sum(path.is_file() for path in news_dir.glob("*.json"))
        if news_dir.exists()
        else 0,
        "markdown": sum(path.is_file() for path in standardized.rglob("*.md"))
        if standardized.exists()
        else 0,
    }


def _normalize_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def _highlight(content: str, query: str) -> str:
    """Escape evidence and highlight query words/phrases safely in HTML."""
    terms = []
    for term in re.findall(r"[\wÀ-ỹ]+(?:\s+[\wÀ-ỹ]+)?", query, flags=re.UNICODE):
        if len(_normalize_for_match(term).replace(" ", "")) >= 3:
            terms.append(term.strip())
    terms = sorted(set(terms), key=len, reverse=True)[:18]
    if not terms:
        return html.escape(content)

    pattern = re.compile("(" + "|".join(re.escape(term) for term in terms) + ")", re.IGNORECASE)
    parts = []
    cursor = 0
    for match in pattern.finditer(content):
        parts.append(html.escape(content[cursor : match.start()]))
        parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
        cursor = match.end()
    parts.append(html.escape(content[cursor:]))
    return "".join(parts)


def _evidence_overlap(content: str, query: str) -> float:
    query_terms = set(_normalize_for_match(query).split())
    content_terms = set(_normalize_for_match(content).split())
    query_terms = {term for term in query_terms if len(term) > 2}
    return len(query_terms & content_terms) / max(len(query_terms), 1)


def render_sources(
    sources: list[dict], query: str = "", expanded: bool = False
) -> None:
    """Render ranked evidence with score, chunk metadata and highlighted text."""
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=expanded):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            source_name = str(
                metadata.get("source")
                or metadata.get("filename")
                or f"Source {index}"
            )
            doc_type = str(metadata.get("type", "unknown"))
            score = source.get("score")
            score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"
            chunk_index = metadata.get("chunk_index", "-")
            overlap = _evidence_overlap(str(source.get("content", "")), query)
            overlap_text = f"{overlap:.0%}" if query else "-"
            method = str(source.get("source", "hybrid"))
            score_kind = "RRF" if method == "hybrid" or "rrf_score" in source else "relevance"
            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-title">[{index}] {html.escape(source_name)}</div>'
                f'<div class="source-meta">Loại: {html.escape(doc_type)} · '
                f'chunk: {html.escape(str(chunk_index))} · phương thức: {html.escape(method)} · '
                f'<span class="score-pill">{score_kind} {score_text}</span> · '
                f'độ khớp từ khóa: <b>{overlap_text}</b></div>',
                unsafe_allow_html=True,
            )
            content = str(source.get("content", "")).strip()
            preview = content[:1200] + ("…" if len(content) > 1200 else "")
            st.markdown(
                f'<div class="evidence">{_highlight(preview, query)}</div></div>',
                unsafe_allow_html=True,
            )
            if index < len(sources):
                st.divider()


MEMORY_WINDOW = 3


def recent_user_questions(exclude: str = "", limit: int = MEMORY_WINDOW) -> list[str]:
    """Return recent user turns in chronological order."""
    questions = []
    for message in reversed(st.session_state.messages):
        if message.get("role") != "user":
            continue
        question = str(message.get("content", "")).strip()
        if question and question != exclude:
            questions.append(question)
        if len(questions) >= limit:
            break
    return list(reversed(questions))


def is_follow_up(query: str, history: list[str]) -> bool:
    """Detect short/anaphoric turns that need conversational context."""
    if not history:
        return False
    normalized = _normalize_for_match(query)
    markers = (
        "cai nay", "phuong thuc nay", "san pham nay", "don nay",
        "the nao", "vay", "the sao", "con ", "them ", "nao khac",
        "bao nhieu nua", "tiep theo", "what about", "how about",
    )
    if any(marker in normalized for marker in markers):
        return True
    short_follow_up_words = {"nay", "do", "vay", "khong", "sao"}
    return len(normalized.split()) <= 6 and bool(short_follow_up_words.intersection(normalized.split()))


def build_retrieval_query(query: str, history: list[str]) -> tuple[str, bool]:
    """Resolve a follow-up against recent questions without polluting new topics."""
    if not is_follow_up(query, history):
        return query, False
    context = "\n".join(f"- {question}" for question in history)
    return (
        "Conversation context (recent user questions):\n"
        f"{context}\n"
        f"Current follow-up question: {query}"
    ), True


def run_rag(
    query: str,
    top_k: int,
    use_reranking: bool,
    score_threshold: float,
    prior_question: str = "",
    history_questions: list[str] | None = None,
) -> dict:
    """Retrieve evidence first, then generate from exactly those chunks."""
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation

    history = list(history_questions or [])
    if not history and prior_question and prior_question != query:
        history = [prior_question]
    retrieval_query, memory_used = build_retrieval_query(query, history)

    sources = retrieve(
        retrieval_query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )
    response = generate_with_citation(
        query,
        context_chunks=sources,
        top_k=top_k,
    )
    response["sources"] = sources
    response["retrieval_query"] = retrieval_query
    response["memory_used"] = memory_used
    response["memory_questions"] = history if memory_used else []
    return response


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption("Tra cứu chính sách thương mại điện tử có trích dẫn nguồn.")

    stats = get_catalog_stats()
    st.subheader("📊 Trạng thái dữ liệu")
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric("Legal", stats["legal"])
    stat_col2.metric("News", stats["news"])
    st.metric("Markdown chuẩn hóa", stats["markdown"])
    st.metric("Lượt hỏi trong phiên", len(recent_user_questions()))

    if stats["markdown"] == 0:
        st.warning("Chưa có dữ liệu Markdown. Hãy hoàn thành Task 3.")
    else:
        st.success("Corpus sẵn sàng cho demo local.")

    st.divider()
    st.subheader("⚙️ Cấu hình truy vấn")
    top_k = st.slider("Số chunks trả về", 1, 10, 5)
    use_reranking = st.checkbox("Bật Hybrid + RRF", value=True)
    score_threshold = st.slider(
        "Ngưỡng fallback PageIndex",
        0.0,
        1.0,
        0.20,
        0.01,
        help="Ngưỡng dense để cân nhắc fallback. Với embedding local hiện tại, 0.20 là mức đã hiệu chỉnh.",
    )

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_query = suggestion

    st.divider()
    if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.caption("Retrieval → RRF/PageIndex fallback → Answer + citation")


st.title("🛒 E-commerce Support RAG Chatbot")
st.caption(
    "Hỏi về chính sách trả hàng, thanh toán, đăng bán sản phẩm, TikTok Shop và thuế."
)

if not st.session_state.messages:
    st.info(
        "Chào bạn! Hãy nhập câu hỏi bên dưới hoặc chọn một câu hỏi gợi ý. "
        "Câu trả lời chỉ dựa trên tài liệu trong corpus của project."
    )
    with st.expander("Kiến trúc demo"):
        st.markdown(
            "1. **Task 5–6:** semantic search và BM25 lexical search\n"
            "2. **Task 7:** gộp kết quả bằng Reciprocal Rank Fusion\n"
            "3. **Task 8:** PageIndex hoặc local fallback khi dense score thấp\n"
            "4. **Task 10:** tạo câu trả lời có citation từ các chunks đã truy hồi"
        )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            retrieval_source = message.get("retrieval_source", "none")
            st.caption(
                f"Retrieval: **{retrieval_source}** · "
                f"{len(message.get('sources', []))} chunks evidence"
            )
            if message.get("memory_used"):
                st.info("🧠 Đã dùng ngữ cảnh các câu hỏi trước để hiểu follow-up.")
            render_sources(
                message.get("sources", []),
                query=message.get("retrieval_query", message.get("content", "")),
            )


typed_query = st.chat_input("Nhập câu hỏi về chính sách/hỗ trợ e-commerce…")
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    query = str(query).strip()
    if not query:
        st.stop()

    history_questions = recent_user_questions(exclude=query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy hồi tài liệu và tạo câu trả lời…"):
            response = {}
            try:
                response = run_rag(
                    query,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    score_threshold=score_threshold,
                    history_questions=history_questions,
                )
                answer = response.get(
                    "answer",
                    "Mình chưa thể tạo câu trả lời từ corpus hiện tại.",
                )
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", "none")
                error_message = None
            except Exception as error:
                answer = (
                    "Mình chưa thể hoàn thành truy vấn. "
                    "Hãy kiểm tra dữ liệu chuẩn hóa và dependencies của project."
                )
                sources = []
                retrieval_source = "error"
                error_message = str(error)

        st.markdown(answer)
        st.caption(
            f"Retrieval: **{retrieval_source}** · {len(sources)} chunks evidence"
        )
        if response.get("memory_used"):
            st.info("🧠 Đã dùng ngữ cảnh các câu hỏi trước để hiểu follow-up.")
        if error_message:
            with st.expander("Chi tiết lỗi"):
                st.code(error_message)
        render_sources(sources, query=response.get("retrieval_query", query), expanded=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "retrieval_query": response.get("retrieval_query", query),
            "memory_used": response.get("memory_used", False),
            "memory_questions": response.get("memory_questions", []),
        }
    )
