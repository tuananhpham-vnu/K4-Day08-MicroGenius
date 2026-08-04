"""Streamlit demo for the E-commerce Support RAG pipeline.

Flow:
    Streamlit UI -> Task 9 retrieval -> Task 10 citation-ready answer

The demo uses the deterministic local retrieval/generation fallback, so it can
run without an external API key.
"""

from __future__ import annotations

import sys
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


def render_sources(sources: list[dict], expanded: bool = False) -> None:
    """Render retrieved evidence with readable metadata and safe previews."""
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
            score_text = (
                f"{float(score):.4f}"
                if isinstance(score, (int, float))
                else "N/A"
            )

            st.markdown(
                f"**[{index}] {source_name}** · loại: [{doc_type}] · điểm: [{score_text}]"
            )
            content = str(source.get("content", "")).strip()
            st.text(content[:700] + ("…" if len(content) > 700 else ""))
            if index < len(sources):
                st.divider()


def previous_user_question(exclude: str = "") -> str:
    """Return the previous user question for follow-up retrieval context."""
    for message in reversed(st.session_state.messages):
        if message.get("role") == "user":
            question = str(message.get("content", ""))
            if question != exclude:
                return question
    return ""


def run_rag(
    query: str,
    top_k: int,
    use_reranking: bool,
    score_threshold: float,
    prior_question: str = "",
) -> dict:
    """Retrieve evidence first, then generate from exactly those chunks."""
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation

    retrieval_query = query
    if prior_question and prior_question != query:
        retrieval_query = (
            f"Previous question: {prior_question}\n"
            f"Follow-up question: {query}"
        )

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
        0.48,
        0.01,
        help="So sánh với cosine score gốc, không phải điểm RRF.",
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
            st.caption(f"Nguồn retrieval: [{retrieval_source}]")
            render_sources(message.get("sources", []))


typed_query = st.chat_input("Nhập câu hỏi về chính sách/hỗ trợ e-commerce…")
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    query = str(query).strip()
    if not query:
        st.stop()

    prior_question = previous_user_question(exclude=query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy hồi tài liệu và tạo câu trả lời…"):
            try:
                response = run_rag(
                    query,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    score_threshold=score_threshold,
                    prior_question=prior_question,
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
        st.caption(f"Nguồn retrieval: [{retrieval_source}]")
        if error_message:
            with st.expander("Chi tiết lỗi"):
                st.code(error_message)
        render_sources(sources, expanded=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )
