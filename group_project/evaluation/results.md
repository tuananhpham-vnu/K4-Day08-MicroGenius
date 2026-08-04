# RAG Evaluation Results

## Framework sử dụng

> **RAGAS** (`ragas==0.1.21`) — 4 metric: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`. Judge LLM: model cấu hình qua `RAGAS_LLM_MODEL` (mặc định `openai/gpt-4o-mini` qua OpenRouter). Embeddings cho `answer_relevancy`: `sentence-transformers/all-MiniLM-L6-v2` chạy local (không tốn quota API).

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-only)* | Δ |
|--------|---------------------------:|-----------------------:|--:|
| Faithfulness | **0.946** | 0.818 | +0.128 |
| Answer Relevance | **0.792** | 0.661 | +0.131 |
| Context Recall | **0.861** | 0.706 | +0.155 |
| Context Precision | **0.972** | 0.804 | +0.168 |
| **Average** | **0.893** | **0.747** | **+0.146** |


---

## A/B Comparison Analysis

### Config A

Hybrid Retrieval kết hợp **Dense Semantic Search** và **BM25 Lexical Search** bằng **Reciprocal Rank Fusion (RRF)**. Sau khi truy hồi, hệ thống sử dụng **Cross-Encoder Reranker** để sắp xếp lại các tài liệu trước khi đưa vào LLM sinh câu trả lời.

### Config B

Dense-only Retrieval chỉ sử dụng **Vector Similarity Search**, không kết hợp BM25 và không sử dụng bước reranking.

### Kết luận

Config A đạt kết quả tốt hơn trên cả bốn metric đánh giá.

Mức cải thiện lớn nhất nằm ở **Context Precision (+0.168)** và **Context Recall (+0.155)**, cho thấy việc kết hợp Hybrid Retrieval với Reranking giúp truy hồi đầy đủ hơn đồng thời giảm số lượng chunk không liên quan. Faithfulness và Answer Relevance cũng được cải thiện nhờ LLM nhận được context chất lượng hơn.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Tại Việt Nam, Dịch Vụ Thanh Toán cho các đơn hàng trên TikTok Shop do đơn vị nào cung cấp? | 0.51 | 0.38 | 0.57 | Retrieval | Retriever không truy hồi được chunk chứa thông tin cần thiết. Context Recall thấp dẫn đến LLM không thể tạo câu trả lời chính xác. |
| 2 | Có bao nhiêu phương pháp tính thuế đối với hộ kinh doanh, cá nhân kinh doanh theo Thông tư 40/2021/TT-BTC? | 0.58 | 0.49 | 0.71 | Generation (Extraction) | Câu trả lời được sinh ra chủ yếu là tiêu đề hoặc mẫu biểu trong văn bản thay vì nội dung trả lời trực tiếp, khiến RAGAS không thể trích xuất statement để đánh giá Faithfulness. |
| 3 | Hoạt động thương mại điện tử của cá nhân có thuộc đối tượng áp dụng của Thông tư 40/2021/TT-BTC không? | 0.62 | 0.54 | 0.76 | Generation (Extraction) | LLM trả về đoạn văn mang tính trích dẫn tài liệu thay vì câu trả lời hoàn chỉnh, gây lỗi trong bước statement extraction của RAGAS. |

---

## Recommendations

### Cải tiến 1

**Action**

- Thay thế phương pháp chunking hiện tại bằng **Statistical Chunking** để tạo các chunk có ranh giới ngữ nghĩa hợp lý hơn.
- Điều chỉnh kích thước chunk và overlap nhằm hạn chế việc cắt mất thông tin.

**Expected impact**

- Tăng **Context Recall**.
- Cải thiện **Faithfulness** nhờ context đầy đủ hơn.

---

### Cải tiến 2

**Action**

- Tối ưu **Hybrid Retrieval** bằng cách điều chỉnh trọng số giữa Semantic Search và BM25 (ví dụ 0.5–0.5 hoặc thực hiện Grid Search).
- Tiếp tục sử dụng Cross-Encoder Reranker để loại bỏ các chunk ít liên quan.

**Expected impact**

- Tăng **Context Recall**.
- Duy trì hoặc cải thiện **Context Precision**.
- Giảm lỗi Retrieval.

---

### Cải tiến 3

**Action**

Bổ sung bước **Query Processing** trước Retrieval:

- Query Rewrite
- Query Decomposition
- Step-back Prompting

Sử dụng **Gemma 3 4B** hoặc một Small Language Model (SLM) để giảm độ trễ của hệ thống.

**Expected impact**

- Cải thiện **Answer Relevance**.
- Tăng **Context Recall**.
- Giúp Retriever tìm đúng tài liệu đối với các câu hỏi dài hoặc nhiều ý.

---

## Summary

Kết quả đánh giá cho thấy hệ thống Hybrid Retrieval kết hợp Reranking đạt hiệu quả cao với:

- Faithfulness: **0.946**
- Context Precision: **0.972**
- Context Recall: **0.861**

Trong bốn metric, **Answer Relevance (0.792)** vẫn là chỉ số thấp nhất và là hướng cần ưu tiên cải thiện trong các phiên bản tiếp theo thông qua tối ưu Query Processing, Chunking và Prompt Engineering.