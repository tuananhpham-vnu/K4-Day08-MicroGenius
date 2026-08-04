# RAG Evaluation Results

Golden dataset: 18 câu hỏi.

## Overall Scores

| Metric | Score |
|--------|-------|
| faithfulness | 0.909 |
| answer_relevancy | 0.093 |
| context_recall | 0.667 |
| context_precision | 0.745 |

## A/B Comparison

| Config | faithfulness | answer_relevancy | context_recall | context_precision |
|--------|-------|-------|-------|-------|
| hybrid_rerank | 0.938 | 0.079 | 0.806 | 0.815 |
| dense_only | 0.874 | 0.093 | 0.806 | 0.733 |

## Worst Performers

| Question | Faithfulness | Answer Relevancy |
|----------|--------------|-------------------|
|  | 0.500 | 0.126 |
|  | 0.667 | 0.000 |
|  | 0.800 | 0.062 |

## Recommendations

- Xem lại các câu hỏi trong bảng "Worst Performers" — điểm faithfulness thấp thường do context thiếu evidence hoặc retrieval trả nhầm chunk.
- So sánh 2 dòng trong bảng A/B: nếu `dense_only` gần bằng `hybrid_rerank`, cân nhắc bỏ bước reranking để giảm latency; nếu chênh lệch lớn, giữ reranking.
