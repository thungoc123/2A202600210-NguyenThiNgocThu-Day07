# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** NguyenThiNgocThu
**Nhóm:** 25
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector (câu) có hướng gần giống nhau, nên nội dung/ngữ nghĩa tương đồng dù từ ngữ có thể khác.

**Ví dụ HIGH similarity:**
- Sentence A: I love learning AI and machine learning.
- Sentence B: I enjoy studying artificial intelligence and ML.
- Tại sao tương đồng: Cùng nói về việc thích học AI, chỉ khác cách diễn đạt (love ~ enjoy, AI ~ artificial intelligence).

**Ví dụ LOW similarity:**
- Sentence A: I love playing football on weekends.
- Sentence B: The stock market crashed yesterday.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (thể thao vs tài chính).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến hướng vector (ngữ nghĩa), nên ít bị ảnh hưởng bởi độ dài/độ lớn vector. Trong khi đó, Euclidean distance phụ thuộc vào độ lớn nên thường kém phù hợp hơn cho so sánh semantic embeddings của văn bản.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính:
> Step size = chunk_size - overlap = 500 - 50 = 450
>
> So chunks = ceil((10000 - 500) / 450) + 1
> = ceil(9500 / 450) + 1
> = ceil(21.11) + 1
> = 22 + 1 = 23
>
> Đáp án: 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, step size mới là 500 - 100 = 400, nên số chunk sẽ tăng vì mỗi lần trượt ngắn hơn. Overlap lớn hơn giúp giữ ngữ cảnh tốt hơn giữa các chunk và giảm mất thông tin ở ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [Luật biển Việt Nam]

**Tại sao nhóm chọn domain này?**
> Thú vị và hữu ích. 

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Luật biển Việt Nam | Chính phủ | 56649 | doc_type=law; legal_code=18/2012/QH13; issuing_body=Quoc_hoi; year=2012; topic=chu_quyen_vung_bien; language=vi |
| 2 | Nghị định 144 | Chính phủ | 12589 | doc_type=decree_amendment; legal_code=144/2018/ND-CP; issuing_body=Chinh_phu; year=2018; topic=van_tai_da_phuong_thuc; language=vi |
| 3 | Nghị định về vận tải đa phương thức | Chính phủ | 56901 | doc_type=decree; legal_code=87/2009/ND-CP; issuing_body=Chinh_phu; year=2009; topic=van_tai_da_phuong_thuc; language=vi |
| 4 | Nghị định về điều kiện kinh doanh | Chính phủ | 26333 | doc_type=decree; legal_code=160/2016/ND-CP; issuing_body=Chinh_phu; year=2016; topic=dieu_kien_kinh_doanh_van_tai_bien; language=vi |
| 5 | Thông tư 66 | Bộ GTVT | 17609 | doc_type=circular; legal_code=66/2014/TT-BGTVT; issuing_body=Bo_GTVT; year=2014; topic=van_tai_hanh_khach_tau_cao_toc; language=vi |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_type | string | law, decree, circular | Lọc đúng loại văn bản pháp lý khi truy vấn |
| legal_code | string | 18/2012/QH13 | Truy hồi chính xác theo số hiệu văn bản |
| year | int | 2012 | Lọc theo mốc thời gian, ưu tiên văn bản mới |
| issuing_body | string | Quoc_hoi, Chinh_phu, Bo_GTVT | Phân biệt thẩm quyền ban hành |
| topic | string | chu_quyen_vung_bien, van_tai_da_phuong_thuc | Tăng độ liên quan theo chủ đề truy vấn |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| luatbienvietnam.md | FixedSizeChunker (`fixed_size`) | 95 | 499.44 | Trung bình (có overlap nhưng dễ cắt giữa ý) |
| luatbienvietnam.md | SentenceChunker (`by_sentences`) | 166 | 255.61 | Tốt (giữ ranh giới câu rõ) |
| luatbienvietnam.md | RecursiveChunker (`recursive`) | 115 | 369.20 | Tốt (cân bằng độ dài và ngữ cảnh) |
| nghidinh144.md | FixedSizeChunker (`fixed_size`) | 22 | 484.50 | Trung bình |
| nghidinh144.md | SentenceChunker (`by_sentences`) | 26 | 367.54 | Tốt |
| nghidinh144.md | RecursiveChunker (`recursive`) | 29 | 328.34 | Tốt |
| thongtu66.md | FixedSizeChunker (`fixed_size`) | 35 | 499.40 | Trung bình |
| thongtu66.md | SentenceChunker (`by_sentences`) | 39 | 401.82 | Tốt |
| thongtu66.md | RecursiveChunker (`recursive`) | 43 | 365.74 | Tốt |

### Strategy Của Tôi

**Loại:** FixedSizeChunker (tuned: `chunk_size=650`, `overlap=120`)

**Mô tả cách hoạt động:**
> FixedSizeChunker chia văn bản theo cửa sổ trượt với kích thước cố định. Mỗi chunk có tối đa 650 ký tự và chồng lấn 120 ký tự để giảm mất mạch nội dung tại ranh giới cắt. Chiến lược này không phụ thuộc dấu câu hoặc cấu trúc heading nên chạy ổn định trên văn bản pháp lý dài và định dạng không đồng nhất. Đổi lại, có thể xuất hiện trường hợp cắt giữa một điều khoản dài.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain luật biển có nhiều văn bản dài, điều khoản nối tiếp nhau và độ dài không đều. Tôi chọn FixedSizeChunker để kiểm soát chi phí xử lý (ít chunk hơn), đồng thời dùng overlap lớn để giữ ngữ cảnh khi truy hồi. Cách này phù hợp khi cần tốc độ và tính ổn định trên tập tài liệu lớn.

**Code snippet (nếu custom):**
```python
# Khong dung custom strategy.
# Su dung FixedSizeChunker(chunk_size=650, overlap=120)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| luatbienvietnam.md | best baseline: RecursiveChunker (`recursive`) | 115 | 369.20 | Tốt (giữ mạch điều khoản rõ hơn) |
| luatbienvietnam.md | **của tôi**: FixedSizeChunker (`chunk_size=650, overlap=120`) | 81 | 646.26 | Khá (nhanh và ít chunk, nhưng đôi khi cắt giữa ý) |
| nghidinh144.md | best baseline: RecursiveChunker (`recursive`) | 29 | 328.34 | Tốt |
| nghidinh144.md | **của tôi**: FixedSizeChunker (`chunk_size=650, overlap=120`) | 18 | 647.17 | Khá |
| thongtu66.md | best baseline: RecursiveChunker (`recursive`) | 43 | 365.74 | Tốt |
| thongtu66.md | **của tôi**: FixedSizeChunker (`chunk_size=650, overlap=120`) | 30 | 641.97 | Khá |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | FixedSizeChunker (650/120) | 8.0 | Nhanh, ít chunk, chi phí embedding thấp | Dễ cắt giữa ý pháp lý dài |
| Thành viên A | RecursiveChunker | 8.8 | Bám ranh giới nội dung tốt, chunk coherent hơn | Nhiều chunk hơn, tốn tài nguyên hơn |
| Thành viên B | SentenceChunker | 7.6 | Dễ đọc theo câu, rõ ranh giới ngữ nghĩa câu | Dễ sinh chunk quá nhỏ hoặc quá dài ở câu pháp lý phức tạp |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Nếu ưu tiên chất lượng retrieval, RecursiveChunker là baseline tốt hơn vì bám ranh giới nội dung tốt hơn trên văn bản pháp lý dài. Nếu ưu tiên tốc độ, bộ nhớ và chi phí embedding, FixedSizeChunker (tuned) phù hợp hơn vì tạo ít chunk hơn đáng kể. Với domain này, tôi chọn FixedSizeChunker cho triển khai thực dụng và có thể kết hợp reranking để bù chất lượng.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi dùng regex `(?<=[.!?])(?:\s+)` để tách ngay sau dấu kết thúc câu. Sau khi tách, tôi strip từng câu và bỏ phần rỗng để tránh chunk trống. Edge case chính là văn bản pháp lý có dòng xuống hàng dày đặc, nên tôi chuẩn hóa khoảng trắng khi ghép câu thành chunk để đảm bảo đầu ra ổn định.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán thử tách theo thứ tự separator ưu tiên (`\n\n`, `\n`, `. `, ` `, rồi fallback ký tự). Base case là khi đoạn hiện tại đã nhỏ hơn hoặc bằng `chunk_size` thì trả luôn, còn nếu hết separator thì cắt cứng theo chiều dài. Cách này giúp giữ ngữ cảnh tự nhiên tốt hơn fixed-size nhưng vẫn có cơ chế đảm bảo không vượt ngưỡng.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` embed từng document/chunk thành vector và lưu cùng metadata vào store nội bộ (hoặc ChromaDB nếu khả dụng). `search` embed câu query rồi tính điểm tương đồng bằng dot product với các vector đã lưu. Kết quả được sort giảm dần theo score và trả về top-k.

**`search_with_filter` + `delete_document`** — approach:
> Tôi filter trước theo metadata (`search_with_filter`) rồi mới tính similarity trên tập đã lọc để giảm nhiễu. `delete_document` duyệt các record theo `doc_id`, loại tất cả chunk thuộc document cần xóa và đồng bộ xóa với Chroma nếu đang bật. Hàm trả về bool để thể hiện có xóa thành công hay không.

### KnowledgeBaseAgent

**`answer`** — approach:
> `answer` truy xuất top-k chunk liên quan từ store, sau đó ghép thành một `Context` block. Prompt gồm 3 phần chính: instruction (chỉ dùng context), question và context để giảm hallucination. Nếu truy xuất rỗng, prompt vẫn trả thông báo thiếu dữ liệu thay vì đoán.

### Test Results

```
========================================== test session starts ===========================================
platform win32 -- Python 3.14.2, pytest-9.0.2
collected 42 items

tests/test_solution.py .......................................... [100%]

=========================================== 42 passed in 0.10s ===========================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Luật biển Việt Nam quy định về lãnh hải và vùng đặc quyền kinh tế. | Văn bản này mô tả lãnh hải và vùng đặc quyền kinh tế của Việt Nam. | high | 0.0683 | Sai |
| 2 | Doanh nghiệp cần giấy phép vận tải đa phương thức quốc tế. | Công ty phải có giấy phép để kinh doanh vận tải đa phương thức xuyên biên giới. | high | -0.0729 | Sai |
| 3 | Tàu khách cao tốc phải niêm yết số điện thoại đường dây nóng. | Hôm nay trời mưa lớn ở khu vực miền núi phía Bắc. | low | 0.1436 | Đúng |
| 4 | Nghị định 160 quy định điều kiện kinh doanh vận tải biển. | Thông tư 66 hướng dẫn vận tải hành khách bằng tàu cao tốc nội thủy. | low | -0.2392 | Đúng |
| 5 | Giấy phép kinh doanh vận tải đa phương thức có hiệu lực 05 năm. | Thời hạn của giấy phép vận tải đa phương thức quốc tế là 5 năm kể từ ngày cấp. | high | 0.1021 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp (5) bất ngờ nhất vì hai câu gần như cùng nghĩa nhưng điểm lại thấp. Điều này cho thấy backend embedding đang dùng trong lúc đo không phản ánh ngữ nghĩa tốt (mock embedding), nên điểm similarity có thể lệch trực giác. Vì vậy cần kiểm tra backend embedding trước khi kết luận về chất lượng semantic retrieval.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | "Đảo" được định nghĩa như thế nào trong Luật Biển Việt Nam? | Đảo là một vùng đất tự nhiên có nước bao bọc, khi thủy triều lên vùng đất này vẫn ở trên mặt nước. |
| 2 | Việc cấp Giấy phép kinh doanh vận tải đa phương thức quốc tế có thời hạn hiệu lực là bao nhiêu năm? | Giấy phép kinh doanh vận tải đa phương thức quốc tế có thời hạn hiệu lực là 05 năm kể từ ngày cấp. |
| 3 | Về điều kiện kinh doanh, Nghị định 144/2018/NĐ-CP đã sửa đổi, bổ sung yêu cầu gì so với Nghị định 87/2009/NĐ-CP? | Nghị định 144/2018 nới lỏng hoặc thay đổi một số thủ tục, điều kiện tài chính và yêu cầu bảo hiểm trách nhiệm nghề nghiệp so với quy định trước đó. |
| 4 | Đối với tàu khách cao tốc, nếu hành khách mang theo vũ khí hoặc chất nổ lên tàu thì có vi phạm không và quy định chi tiết là gì? | Hành khách không được mang vũ khí, chất cháy nổ lên tàu; vi phạm sẽ bị từ chối vận chuyển và xử lý theo quy định pháp luật. |
| 5 | Hãy tìm các điều kiện kinh doanh dịch vụ lai dắt tàu biển (chỉ trong văn bản loại nghị định). | Doanh nghiệp phải thành lập hợp pháp, đáp ứng điều kiện về phương tiện, nhân lực, an toàn hàng hải và nghĩa vụ pháp lý liên quan theo quy định nghị định. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | "Đảo" được định nghĩa như thế nào trong Luật Biển Việt Nam? | Top-1 nằm trong `luatbienvietnam.md`, chủ yếu phần mở đầu văn bản luật biển. | 0.7078 | Partially | Hệ thống xác định đúng tài liệu nguồn; cần chunk nhỏ hơn hoặc top-k cao hơn để lấy đúng điều khoản định nghĩa "đảo". |
| 2 | Việc cấp Giấy phép kinh doanh vận tải đa phương thức quốc tế có thời hạn hiệu lực là bao nhiêu năm? | Top-1 từ `nghidinh144.md`, chứa đoạn về giấy phép vận tải đa phương thức quốc tế. | 0.7510 | Yes | Trả lời được thời hạn hiệu lực giấy phép là 05 năm kể từ ngày cấp. |
| 3 | Nghị định 144/2018/NĐ-CP sửa đổi gì so với Nghị định 87/2009/NĐ-CP? | Top-1 từ `nghidinh144.md`, phần mở đầu/sửa đổi bổ sung của nghị định. | 0.7976 | Yes | Tóm tắt rằng Nghị định 144 điều chỉnh và bãi bỏ một số quy định cũ, thay đổi điều kiện/thủ tục liên quan vận tải đa phương thức. |
| 4 | Hành khách mang vũ khí hoặc chất nổ lên tàu khách cao tốc có vi phạm không? | Top-1 từ `thongtu66.md`, thuộc phần quy định vận tải hành khách tàu cao tốc. | 0.7529 | Yes | Trả lời là vi phạm quy định an toàn, có thể bị từ chối vận chuyển và xử lý theo pháp luật. |
| 5 | Điều kiện kinh doanh dịch vụ lai dắt tàu biển (lọc văn bản nghị định) | Top-1 từ `nghidinhvedieukienkinhdoanh.md`, có đoạn điều kiện kinh doanh dịch vụ hàng hải. | 0.7404 | Yes | Nêu các điều kiện chính: doanh nghiệp hợp pháp, điều kiện phương tiện, nhân lực và yêu cầu an toàn/pháp lý liên quan. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được rằng chunking theo ranh giới nội dung (điều/chương) thường giúp retrieval văn bản luật ổn định hơn chunking cố định. Thành viên dùng RecursiveChunker cho kết quả coherent hơn ở các câu hỏi yêu cầu trích đúng điều khoản. Điều này giúp tôi nhìn rõ trade-off giữa tốc độ và độ chính xác ngữ cảnh.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác dùng metadata filter rất chặt theo loại văn bản (`law`, `decree`, `circular`) trước khi search nên giảm nhiễu đáng kể. Họ cũng log thêm hit@k và thời gian query để đánh giá không chỉ đúng/sai mà cả hiệu năng. Cách trình bày metric đó rất hữu ích cho việc so sánh strategy.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tách tài liệu pháp lý theo đơn vị Điều/Khoản trước khi embedding thay vì chỉ dùng fixed-size. Tôi cũng sẽ chuẩn hóa metadata sâu hơn (chapter, article, clause) để filter tốt hơn ở các câu hỏi pháp lý cụ thể. Ngoài ra, tôi sẽ benchmark thêm model multilingual để tăng chất lượng semantic retrieval tiếng Việt.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **92 / 100 (quy đổi từ 83/90)** |
