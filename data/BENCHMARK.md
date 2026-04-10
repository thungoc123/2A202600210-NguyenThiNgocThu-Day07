# Danh sách Câu hỏi Benchmark (Domain: Luật & Vận tải biển VN)

Dưới đây là 5 câu hỏi Benchmark được thiết kế theo độ khó tăng dần để kiểm tra các chiến lược chunking và retrieval trong hệ thống RAG.

## 1. Bảng tóm tắt câu hỏi (Dùng cho Section 6 của Báo cáo)

| # | Câu hỏi | Trả lời kỳ vọng (Gold Answer) | Loại câu hỏi | File tài liệu chứa đáp án | Có dùng Filter không? |
|---|---|---|---|---|---|
| Q1 | "Đảo" được định nghĩa như thế nào trong Luật Biển Việt Nam? | Đảo là một vùng đất tự nhiên có nước bao bọc, khi thủy triều lên vùng đất này vẫn ở trên mặt nước. | Factual (Đơn giản) | `LUAT_BIEN_VIET_NAM.md` | Không |
| Q2 | Việc cấp Giấy phép kinh doanh vận tải đa phương thức quốc tế có thời hạn hiệu lực là bao nhiêu năm? | Giấy phép kinh doanh vận tải đa phương thức quốc tế có thời hạn hiệu lực là 05 năm kể từ ngày cấp. | Nội dung kéo dài | `NGHỊ ĐỊNH VỀ VẬN TẢI ĐA PHƯƠNG THỨC.md` | Không |
| Q3 | Về điều kiện kinh doanh, Nghị định 144/2018/NĐ-CP đã sửa đổi, bổ sung yêu cầu gì so với Nghị định 87/2009/NĐ-CP? | Nghị định 144/2018 nới lỏng hoặc thay đổi các thủ tục, điều kiện tài chính hoặc chứng chỉ bảo hiểm trách nhiệm nghề nghiệp so với bản cũ 2009. | Multi-doc (Tổng hợp) | `Nghị định 144_2018_NĐ-CP.md` & `NGHỊ ĐỊNH VỀ VẬN TẢI ĐA PHƯƠNG THỨC.md` | Không |
| Q4 | Đối với tàu khách cao tốc, nếu hành khách mang theo vũ khí hoặc chất nổ lên tàu thì có vi phạm không và quy định chi tiết là gì? | Hành khách tuyệt đối không được mang vũ khí, chất cháy nổ lên tàu khách cao tốc, vi phạm sẽ bị cấm lên tàu và xử lý theo quy định pháp luật. | Suy luận | `Thông tư 66_2014_TT-BGTVT.md` | Không |
| Q5 | Hãy tìm các điều kiện kinh doanh dịch vụ lai dắt tàu biển. (Chỉ tìm trong văn bản loại "nghi_dinh") | Doanh nghiệp phải được thành lập hợp pháp, đáp ứng điều kiện về cơ sở vật chất, thuyền viên và ký quỹ hoặc bảo hiểm trách nhiệm pháp lý. | Metadata Filter | `NGHỊ ĐỊNH  VỀ ĐIỀU KIỆN KINH DOANH VẬN TẢI BIỂN.md` | **Có** (`{"doc_type": "nghi_dinh"}`) |

---

## 2. Dữ liệu chuẩn bị cho Python (Dùng cho `evaluation.py`)

Bạn có thể copy đoạn code sau vào file `evaluation.py` của mình để bắt đầu test vòng lặp benchmark tự động:

```python
BENCHMARK = [
    {
        "id": "Q1",
        "question": "\"Đảo\" được định nghĩa như thế nào trong Luật Biển Việt Nam?",
        "gold_answer": "Đảo là một vùng đất tự nhiên có nước bao bọc, khi thủy triều lên vùng đất này vẫn ở trên mặt nước.",
        "relevant_doc_sources": ["LUAT_BIEN_VIET_NAM.md"],
        "requires_filter": False
    },
    {
        "id": "Q2",
        "question": "Dựa vào Nghị định về vận tải đa phương thức (87/2009/NĐ-CP), việc cấp Giấy phép kinh doanh vận tải đa phương thức quốc tế có thời hạn hiệu lực là bao nhiêu năm?",
        "gold_answer": "Giấy phép kinh doanh vận tải đa phương thức quốc tế có thời hạn hiệu lực là 05 năm.",
        "relevant_doc_sources": ["NGHỊ ĐỊNH VỀ VẬN TẢI ĐA PHƯƠNG THỨC.md"],
        "requires_filter": False
    },
    {
        "id": "Q3",
        "question": "Để kinh doanh vận tải đa phương thức quốc tế, điều kiện đăng ký theo Nghị định 144/2018/NĐ-CP có gì khác với quy định trước đó tại Nghị định 87/2009/NĐ-CP?",
        "gold_answer": "Nghị định 144/2018 nới lỏng và bãi bỏ một số thủ tục, sửa đổi điều kiện về tài sản, bảo hiểm nghề nghiệp so với Nghị định 87/2009.",
        "relevant_doc_sources": ["NGHỊ ĐỊNH VỀ VẬN TẢI ĐA PHƯƠNG THỨC.md", "Nghị định 144_2018_NĐ-CP.md"],
        "requires_filter": False
    },
    {
        "id": "Q4",
        "question": "Đối với hành khách đi tàu khách cao tốc nội thủy, việc mang theo hàng hóa nguy hiểm như chất nổ, vũ khí bị xử lý như thế nào dựa trên các quy định hiện hành?",
        "gold_answer": "Hành khách bị nghiêm cấm mang theo vũ khí, chất cháy, chất nổ. Thuyền trưởng có quyền từ chối vận chuyển và giao cho cơ quan có thẩm quyền xử lý.",
        "relevant_doc_sources": ["Thông tư 66_2014_TT-BGTVT.md"],
        "requires_filter": False
    },
    {
        "id": "Q5",
        "question": "Hãy nêu chi tiết các điều kiện bắt buộc để kinh doanh dịch vụ lai dắt tàu biển tại Việt Nam.",
        "gold_answer": "Phải là doanh nghiệp thành lập hợp pháp, có tàu lai dắt chuyên dụng, thuyền viên có bằng cấp/chứng chỉ tương ứng và tuân thủ các quy định về an toàn hàng hải.",
        "relevant_doc_sources": ["NGHỊ ĐỊNH  VỀ ĐIỀU KIỆN KINH DOANH VẬN TẢI BIỂN.md"],
        "requires_filter": True,
        "filter_dict": {"doc_type": "nghi_dinh"}
    }
]
```
