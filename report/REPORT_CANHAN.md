# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Leduybao (2A202602749)
**Nhóm:** G20
**Ngày:** 2026-09-20
**Chiến lược cá nhân:** Sentence chunk (`SentenceChunker`, `max_sentences_per_chunk=3`)

> Corpus dùng chung: `data/shopee/` (9 file, xem `REPORT_NHOM.md`). Embedder chạy thực tế: `_mock_embed` (mặc định, không cài thêm model).

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**
> Hai vector embedding cùng hướng, tức hai đoạn văn bản có ý nghĩa gần nhau trong không gian embedding (gần 1.0 = rất giống, 0 = trực giao/không liên quan, -1 = ngược nghĩa).

**Ví dụ có độ tương tự CAO:**
- Câu A: Người Mua có 15 ngày để yêu cầu trả hàng/hoàn tiền.
- Câu B: Người Mua có thể yêu cầu trả hàng trong vòng 15 ngày sau khi nhận hàng.
- Tại sao tương đồng: cùng ý, chỉ khác cách diễn đạt (thời hạn 15 ngày + hành động trả hàng).

**Ví dụ có độ tương tự THẤP:**
- Câu A: Người Bán phải cung cấp bằng chứng trong 24 giờ.
- Câu B: Thời tiết hôm nay rất đẹp và nắng.
- Tại sao khác: một câu về nghĩa vụ chứng minh, một câu về thời tiết — không chung chủ đề/từ khóa.

**Tại sao cosine được ưu tiên hơn Euclidean cho text embeddings?**
> Cosine chỉ so hướng vector (ngữ nghĩa), bỏ qua độ dài vector (độ dài văn bản/tần suất từ). Euclidean bị ảnh hưởng bởi norm nên chunk dài/ngắn khó so sánh công bằng.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `ceil((10000-50)/(500-50)) = ceil(9950/450) = ceil(22.11) = 23 chunks`.
> Đáp án: **23 chunks**.

**Nếu overlap tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `ceil((10000-100)/(500-100)) = ceil(9900/400) = ceil(24.75) = 25 chunks` — tăng 23 lên 25. Muốn overlap lớn hơn để giữ ngữ cảnh ở biên chunk (đặc biệt điều khoản ghi số ngày, điều kiện), tránh cắt mất ý nối giữa 2 chunk.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng `re.split(r"(?<=[.!?])\s+", text.strip())` để tách theo `. `, `! `, `? ` và `.\n` (vì `\s+` bao cả `\n`), giữ dấu câu nhờ lookbehind, `strip()` và loại câu rỗng, rồi gom tối đa 3 câu/chunk bằng `" ".join()`. Edge case: chuỗi rỗng/khoảng trắng -> `[]`; văn bản không có dấu câu -> 1 chunk nguyên.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `chunk()` kiểm tra `len <= chunk_size` thì trả về ngay, `separators=[]` thì fallback cắt lát cố định. `_split()` lấy separator đầu tiên, nếu không có trong text thì đệ quy với separators còn lại; nếu có thì `split()`, gom tham lam (greedy) các part sao cho `<= chunk_size`, part nào vẫn quá lớn thì đệ quy tiếp với separators ưu tiên thấp hơn. Base case là `len <= chunk_size` hoặc hết separators / gặp `""` thì cắt theo ký tự.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `_make_record()` chuẩn hoá mỗi doc thành `{id: doc_id_index, content, metadata{...doc.metadata, doc_id}, embedding=_embedding_fn(content)}`, append vào `self._store` (đồng thời `collection.add` nếu có chroma). `_search_records()` embed query, tính dot-product với mọi record, sort giảm dần, cắt `top_k`. Vì `_mock_embed` đã chuẩn hoá norm nên dot-product tương đương cosine.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc trước (pre-filter) theo `metadata_filter` bằng `all(record.metadata[k]==v)`, rồi mới similarity search trên tập đã lọc — đúng yêu cầu L3B (`audience`). `delete_document(doc_id)` lọc bỏ mọi record có `metadata.doc_id==doc_id` (và xoá ids tương ứng trong chroma nếu có), trả `True/False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `store.search(question, top_k)` lấy context, nối bằng `\n\n` thành `Context:\n...\n\nQuestion:...\nAnswer based only on the context above.`, rồi gọi `llm_fn(prompt)` và trả nguyên văn. Không tự sinh câu trả lời ngoài context (grounding).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
42 passed in 0.12s (Python 3.13.3 venv, pytest-9.1.1)
TestProjectStructure, TestClassBasedInterfaces, TestFixedSizeChunker (7),
TestSentenceChunker (4), TestRecursiveChunker (4), TestEmbeddingStore (8),
TestKnowledgeBaseAgent (2), TestComputeSimilarity (4),
TestCompareChunkingStrategies (3), TestEmbeddingStoreSearchWithFilter (3),
TestEmbeddingStoreDeleteDocument (3) — tất cả PASSED.
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Chạy `compute_similarity(_mock_embed(a), _mock_embed(b))`. Lưu ý `_mock_embed` là hash MD5 + PRNG, **không mang ngữ nghĩa**, nên điểm thực tế đều quanh 0.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người Mua có 15 ngày để yêu cầu trả hàng/hoàn tiền. | Người Mua có thể yêu cầu trả hàng trong vòng 15 ngày sau khi nhận hàng. | cao | -0.0442 | Sai |
| 2 | Người Bán phải cung cấp bằng chứng trong 24 giờ. | Thời tiết hôm nay rất đẹp và nắng. | thấp | -0.0462 | Đúng (gần 0) |
| 3 | Shopee Mall cho phép trả hàng trong 6 ngày. | Shopee Mall yêu cầu gửi trả sản phẩm trong 6 ngày lịch. | cao | -0.0524 | Sai |
| 4 | Bảo hành do Người Bán tiếp nhận. | Người Bán chịu trách nhiệm tiếp nhận bảo hành. | cao | -0.1898 | Sai |
| 5 | Tranh chấp được giải quyết trong 7 ngày làm việc. | Thực phẩm tươi sống chỉ có 24 giờ để khiếu nại. | thấp | -0.0132 | Đúng (gần 0) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 1 và 4 gần như đồng nghĩa nhưng điểm vẫn âm/gần 0. Điều này chứng tỏ `_mock_embed` không biểu diễn ngữ nghĩa mà chỉ là vector ngẫu nhiên tất định từ hash — muốn điểm cosine phản ánh nghĩa thật phải dùng embedder thật (local multilingual / OpenAI / Gemini).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Cấu hình chạy: `SentenceChunker(max_sentences_per_chunk=3)` trên 9 file `data/shopee` = **224 chunks**, `EmbeddingStore` mock, `top_k=3`. Bảng dưới ghi kết quả `search_with_filter` (cấu hình thi chính).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người Mua có bao lâu để gửi yêu cầu trả hàng/hoàn tiền sau khi đơn được giao thành công? (filter `audience=buyer`) | `warranty-general-buyer.md` — chunk khuyến cáo kiểm tra chính sách bảo hành (score 0.2589) | 0.2589 | Không — gold nằm ở `return-refund-policy.md` (đứng #2 khi không filter, score 0.2434) | Agent chỉ tóm lại context sai (khuyến cáo bảo hành), không ra được 15 ngày/24h |
| 2 | Sau khi yêu cầu Shopee Mall được chấp thuận, Người Mua có bao lâu để gửi trả sản phẩm? (filter `audience=buyer`) | `extended-warranty-insurance.md` — chunk hoá đơn y tế (0.2480); gold `shopee-mall-buyer.md` đứng #3 (0.2158) | 0.2480 | Một nửa — đúng file ở top-3 nhưng không top-1 | Agent nhiễu bởi 2 chunk bảo hiểm y tế, trả lời thiếu số 6 ngày lịch |
| 3 | Khi Shopee yêu cầu bằng chứng cho yêu cầu Shopee Mall, Người Bán phải cung cấp trong bao lâu? (filter `audience=seller` bắt buộc) | `warranty-general-seller.md` — chunk Nghị định 52 (0.2219); gold `shopee-mall-seller.md` đứng #3 (0.2045) | 0.2219 | Một nửa — đúng nhóm seller, đúng file ở top-3 | Agent không trích được mốc 24 giờ vì chunk top-1 sai |
| 4 | Ai chịu trách nhiệm tiếp nhận bảo hành cho Người Mua? (filter `audience=seller`) | `shopee-mall-seller.md` — chunk phí Shopee Mall (0.2178); gold `warranty-general-seller.md` đứng #2/#3 | 0.2178 | Một nửa — đúng nhóm seller, đúng file ở top-3 | Agent nói chung chung về nghĩa vụ seller, thiếu vế Shopee không trực tiếp bảo hành |
| 5 | Tranh chấp không phải trả hàng/hoàn tiền, Shopee ra hướng giải quyết trong bao lâu? (không filter) | `warranty-general-buyer.md` — chunk đăng ký thành viên (0.3081), cả top-3 đều không phải `dispute-resolution.md` | 0.3081 | Không | Agent bịa theo context đăng ký thành viên, sai mốc 7 ngày làm việc |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 3 / 5 (câu 2,3,4 đúng file nguồn ở top-3 khi dùng filter; câu 1,5 trượt). Nếu tính chunk chứa đúng mệnh đề gold (15 ngày, 6 ngày, 24h, 7 ngày) thì **0-1/5** — đây là failure case phải thừa nhận.

**Failure case chính:** mock embedder cho điểm 0.20-0.31 dàn đều, không phân biệt ngữ nghĩa (xem mục 4), nên top-1 gần như ngẫu nhiên. Sentence chunk giữ được câu hoàn chỉnh (coherence tốt) nhưng không cứu được ranking. Filter `audience` giúp câu 2,3,4 kéo đúng nhóm seller/buyer vào top-3, nhưng lại làm mất gold ở câu 1 vì file gold `audience=all` không khớp `buyer` tuyệt đối — cần filter dạng `in [buyer, all]` thay vì `==`.
**Đề xuất:** chuyển sang `LocalEmbedder` đa ngữ + viết filter hỗ trợ `all`, tăng `max_sentences_per_chunk` hoặc chunk theo heading cho các điều khoản dài.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> So với bạn làm heading/section chunk, Sentence chunk của tôi coherent hơn Fixed nhưng thua ở các điều khoản đánh số (3.1, 4.3) vì gom 3 câu làm loãng từ khóa số ngày. Bài học: với chính sách TMĐT nên chunk theo mục + giữ metadata `policy_type` để filter trước.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 |
| **Tổng phần cá nhân** | **56 / 60** |
