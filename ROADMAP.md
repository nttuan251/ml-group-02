# Roadmap đồ án: L2P — Learning to Prompt for Continual Learning

> Paper: Z. Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022 (A*).
> Nhóm 3 người · Compute: 3 tài khoản Kaggle T4 x2 · Tracking: W&B public project.
> **Chỉ còn 6 tuần** (26/10 → 05/12/2026). Mọi quyết định ưu tiên việc nộp đúng hạn với chất lượng cao, thay vì mở rộng scope.

## 0. Định vị (theo career strategy)

- **Tier 1.** Học sâu về continual learning, ViT, prompt tuning; có thêm phần engineering thật: pipeline tái lập được, W&B, config, seed, feature cache, logging thiết kế trước.
- **Góc research:** các điểm yếu trong `docs/critique.md` chính là các hướng follow-up của L2P.
- **Góc MLOps:** experiment tracking, reproducibility, quản lý GPU quota. Đây là kỹ năng cho MLE/MLOps.
- **Không làm:** không viết lại ViT (dùng `timm`); không làm UI/demo; không mở rộng scope nếu gate chưa đạt.

## 1. Mốc nộp (Moodle)

| Mốc | Sản phẩm | Trọng số | Deadline |
|---|---|---|---|
| Selection | Form đăng ký L2P | Điều kiện | ASAP |
| Repo | Link GitHub **public** (vẫn commit tiếp được đến hạn poster) | — | **T7 07/11 23:59** |
| Deck | `ID.pptx` (ID = số nhóm trên sheet), 20–40 slide, notes ≤500 từ/slide, <20 MB | 45% | **T7 21/11 23:59** |
| Poster | `ID.pptx`, 1 slide A2 dọc | 50% | **T7 05/12 23:59** |
| Vấn đáp | Poster workshop | — | Chưa có ngày |
| Liên tục | Repo, W&B, định dạng | 5% | Suốt kỳ |

Nhóm tự đặt hạn chót nội bộ **sớm hơn 2 ngày** (freeze T5) để có thời gian sửa lỗi định dạng và dung lượng file.

## 2. Timeline 6 tuần

**Scope Core+** (chi tiết trong `docs/experiments.md`): R1 Split CIFAR-100 headline, R2 ablation, R3 Gaussian CIFAR-100, R4 sweep tự thiết kế. 5-datasets là stretch.

**"Instrument once":** chỉ có 2 tuần giữa deck và poster, nên logging phục vụ empirics phải được viết ngay từ Tuần 2 (acc matrix, histogram chọn prompt, norm của head, latency...). Khi đó các run tái lập đồng thời sinh luôn dữ liệu cho critique.

| Tuần | A — Method | B — Data & Protocol | C — Training & Experiments | Mốc |
|---|---|---|---|---|
| **1** (26/10–01/11) | Đọc paper + derivation; khung prompt pool | Tạo repo public, W&B, Kaggle; data CIFAR-100 + split; **smoke test throughput** | Đọc paper + derivation; khung train loop + logging | **G1 (T5 29/10)** |
| **2** (02/11–08/11) | Hoàn thiện selection + surrogate loss + diversified + test | Metric, Gaussian schedule, feature cache, README | Loop end-to-end, class mask, AMP, logging "instrument once"; run R1 đầu tiên | **Nộp repo T7 07/11** · **G2 (CN 08/11)** |
| **3** (09/11–15/11) | R2 ablation; nháp slide phương pháp; viết giả thuyết L3/L4 | R3 Gaussian; nháp slide protocol; viết giả thuyết L2 | R1 đủ 3 seed + baseline; E1 NCM (rẻ); viết giả thuyết L1 | **G3 (CN 15/11)** |
| **4** (16/11–21/11) | R4 sweep; hoàn thiện slide của mình | Bảng tái lập + link W&B; kiểm tra quy định deck | Phân tích sai lệch; ghép deck | **Freeze T5 19/11 → nộp deck T7 21/11** |
| **5** (22/11–29/11) | E4, E5 (phần lớn lấy từ log có sẵn) + E6 | E3 (bỏ class mask), E7 (thứ tự task) | E2 đổi backbone (đắt nhất → chạy ngay đầu tuần) | **Chọn 2 điểm yếu (CN 29/11)** |
| **6** (30/11–05/12) | Panel poster của mình | Panel poster + README cho điểm yếu thứ 3 | Thiết kế, ghép poster | **Freeze T5 03/12 → nộp poster T7 05/12** |
| Sau 05/12 | Tổng duyệt vấn đáp | | | |

Mỗi người viết speaker notes **cùng lúc** với slide của mình, không để dồn đến cuối.

### Decision gates
- **G1 (T5 29/10):** smoke test đo được thời gian 1 run Split CIFAR-100. Nếu >2h/run: cache query, giảm tần suất eval, dùng grad accumulation.
- **G2 (CN 08/11):** core chạy end-to-end ít nhất 2 task, và metric đã được test. Nếu chưa đạt, cả nhóm dồn vào core 2 ngày; R2/R3 lùi lại.
- **G3 (CN 15/11):** R1 lệch so với paper ≤3 điểm → làm tiếp R4 (và stretch nếu dư người). Lệch >3 điểm → **đóng băng scope**, dồn sức tìm nguyên nhân sai lệch (đây cũng là nội dung có giá trị cho deck).

## 3. Chia việc: lát dọc + review chéo

Nguyên tắc: **mỗi người sở hữu trọn một chuỗi** lý thuyết → code → thí nghiệm → slide → panel poster. Đồng thời, mỗi người review và viết test cho module của một người khác, nên **mọi module đều có 2 người hiểu**.

| | **A — Method** | **B — Data & Protocol** | **C — Training & Experiments** |
|---|---|---|---|
| Module core tự viết | Prompt pool, key–query top-N, surrogate loss, diversified selection | Split theo task, class order theo seed, Gaussian schedule, metric Avg Acc / Forgetting, feature cache | Train/eval loop theo task, head + class mask, AMP, W&B logging, config |
| Review chéo (phải giải thích lại được) | module của C | module của A | module của B |
| Viết unit test cho module **của người khác** | metric (của B) | selection (của A) | loop trên toy data (của C) |
| Thí nghiệm tái lập | R2 ablation, R4 sweep | R3 Gaussian | R1 headline + phân tích sai lệch |
| Phần deck | Phương pháp (pool, query, loss), độ phức tạp | Dataset, metric, protocol, bảng R3, bảng ánh xạ W&B | Bối cảnh, đóng góp, kết quả R1, sai lệch |
| Hạn chế sở hữu | **L3/L4** prompt collapse, drift (E4, E5) | **L2** thông tin task ngầm (E3, E7) | **L1** phụ thuộc pretrain (E1, E2) |
| Việc phụ | E6 chi phí | Quản lý quota; file `ID.pptx` đúng quy định; README | Thiết kế poster |
| Tài khoản Kaggle | của A | của B | của C |

- **Gán người thật:** người code mạnh nhất → A (phần khó nhất về code); người viết/lý thuyết tốt → C (bối cảnh, headline, poster); người cẩn thận, có tổ chức → B (protocol, metric, README, quota). Điền tên vào đây: A = `____`, B = `____`, C = `____`.
- Poster chỉ chứa **2 trong 3** hạn chế (chọn vào CN 29/11 dựa trên kết quả); hạn chế còn lại đưa vào README kèm bằng chứng. Cả 3 lát đều được tính là công sức.
- Tải việc đã được cân bằng: A nặng ở code Tuần 1–2, C nặng ở run Tuần 3–4, B nặng ở hạ tầng Tuần 1–2 và giấy tờ Tuần 4/6.

## 4. Quy tắc làm việc (bắt buộc — vi phạm có thể 0 điểm)

1. **Tự cài đặt core từ mô tả paper.** Code gốc chỉ để đọc/chạy/so sánh, **không copy**. Thư viện phụ (timm, torchvision, datasets) được dùng, ghi rõ trong README.
2. Không dán text/hình của paper vào deck/poster; vẽ lại toàn bộ.
3. Mọi số liệu ↔ một W&B run public. Không có link = không có căn cứ.
4. Seed cố định và được log; config của mọi run được lưu lên W&B.
5. Commit thường xuyên qua PR — lịch sử commit là bằng chứng tiến độ.
6. **Mỗi người chỉ dùng tài khoản Kaggle của chính mình** (điều khoản Kaggle cấm tài khoản phụ).

## 5. Ngân sách compute: 3 tài khoản × 30h/tuần

Thời gian chạy dưới đây là **ước tính, chưa đo**. Kiểm chứng ở G1.

1. Quota tính theo thời gian session → **mỗi session luôn chạy 2 run song song**, mỗi run một GPU.
2. Debug bằng CPU hoặc subset nhỏ; chỉ bật GPU khi code đã chạy đúng.
3. Run dài thì dùng "Save & Run All (Commit)".
4. Mỗi tài khoản đặt trần 24h/tuần (giữ 6h dự phòng) → khoảng 72h/tuần cho cả nhóm.
5. Gắn tag `account:<tên>` vào mỗi W&B run để theo dõi quota.

| Tuần | Run chính | Ước tính session-giờ | Tài khoản |
|---|---|---|---|
| 1 | Smoke test, cache CIFAR-100 | ~3 | B |
| 2 | Debug, run R1 đầu tiên | ~6 | A, C |
| 3 | R1 (3 seed + 2 baseline), R2 (2 biến thể × 1–3 seed), R3 (3 seed), E1 | ~15 | C, A, B |
| 4 | R4 sweep (~8 cấu hình), chạy lại do sai lệch | ~10 | A, C |
| 5 | E2 (đổi backbone), E3, E7, E5 | ~20 | C, B, A |
| 6 | Bổ sung seed / sửa lỗi | ~5 | dự phòng |
| **Tổng** | | **~60h** | trên tổng sức chứa ~430h |

GPU dư nhiều; **nút thắt là thời gian của 3 người**.

## 6. Rủi ro & phương án

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Form bị reject / nhóm khác lấy L2P | Sheet ghi Rejected | Dự phòng: Loss of Plasticity (Nature 2024) hoặc CoTTA (CVPR 2022) — xem `docs/paper_selection.md` |
| Core trễ | G2 không đạt | Cả nhóm dồn vào core; R2/R3 bỏ bớt seed |
| R1 lệch nhiều so với 83.83% | G3: >3 điểm | Đóng băng scope; kiểm tra checkpoint backbone, class mask, normalization [0,1], class order; mỗi nguyên nhân phải được kiểm chứng bằng một run riêng |
| Không đủ thời gian empirics (chỉ 2 tuần) | Tuần 5 chưa có dữ liệu | Logging "instrument once" từ Tuần 2; viết giả thuyết từ Tuần 3; E2 chạy ngay đầu Tuần 5 |
| File nộp sai quy định | >20 MB, sai tên, thiếu notes | B kiểm tra theo checklist §7 ở ngày freeze |
| Thành viên không hiểu phần người khác | Random viva lúng túng | Cơ chế ở §8 |

## 7. Checklist hướng tới điểm tối đa

Không ai đảm bảo được 10 điểm: điểm do giám khảo chấm, và phần vấn đáp phụ thuộc vào từng người. Checklist này bám sát từng tiêu chí trong đề, để nhóm không mất điểm ở chỗ có thể kiểm soát được.

**Deck (45%) — đề: "deck chỉ diễn lại paper được ít điểm; deck thể hiện đã hiểu, tự vẽ lại và kiểm chứng sẽ đạt"**
- [ ] Ví dụ thực tế của *nhóm* cho bài toán; ≥2 hướng tiền nhiệm, mỗi hướng có ý tưởng / hạn chế / hệ quả.
- [ ] Đóng góp được phân loại lý thuyết / kỹ thuật / thực nghiệm, kèm "trước đây chưa làm được gì".
- [ ] Ký hiệu thống nhất của nhóm; giải thích **mọi biến** trong mọi công thức.
- [ ] Pipeline tự vẽ; nói thẳng rằng paper **không có định lý**, rồi trình bày trực giác (key–query giống k-means/VQ).
- [ ] Độ phức tạp thời gian/bộ nhớ so với ER/EWC, có số cụ thể (46K tham số; 2 forward; 222 token).
- [ ] Bảng tái lập: mean ± std qua 3 seed, sai lệch tương đối; **mọi số có link W&B**.
- [ ] Mỗi sai lệch có nguyên nhân được kiểm chứng bằng một run riêng, không chỉ phỏng đoán. Nếu tìm ra chỗ code gốc khác paper, đó là phần giá trị nhất.
- [ ] Nêu rõ scope Core và lý do chọn (đề yêu cầu).
- [ ] 20–40 slide; **mọi slide** có speaker notes (≤500 từ); trích dẫn IEEE ở chân slide; không có slide References; không có code thô; file <20 MB; tên file `ID.pptx`.

**Poster (50%) — đề: "nêu điểm yếu thật, lập luận lý thuyết, kiểm chứng bằng thí nghiệm thiết kế đúng"**
- [ ] ≥3 điểm mạnh có bằng chứng; ≥2 hạn chế **lý thuyết**; ≥2 hướng phát triển, mỗi hướng gắn một hạn chế.
- [ ] Mỗi hạn chế đi theo chuỗi: giả định → dự đoán khi giả định bị vi phạm → thí nghiệm kiểm định đúng dự đoán đó → kết quả có error bar.
- [ ] Có ít nhất một thí nghiệm mà kết quả *có thể* bác bỏ giả thuyết của nhóm.
- [ ] Điểm nào đã được follow-up (DualPrompt, CODA-Prompt...) chỉ ra thì phải trích dẫn.
- [ ] Lý thuyết đặt cạnh bằng chứng; đúng 2 điểm yếu trên poster, phần còn lại ghi trong README; QR code tới W&B.
- [ ] 1 slide A2 dọc, tên file `ID.pptx`.

**Repo + W&B (5%)**
- [ ] Commit đều đặn qua PR có review; README có bảng ánh xạ ID ↔ script ↔ W&B ↔ kết quả.
- [ ] W&B public; mỗi run có tag theo lab, config và seed đầy đủ.

## 8. Nhịp làm việc & cơ chế "ai cũng hiểu"

1. **Derivation trên giấy (Tuần 1):** mỗi người tự viết shape của tensor qua từng bước forward (ảnh 224×224 → 196 patch × 768 → +CLS → +25 prompt token = 222 token → avgpool 25 vị trí prompt → logits 100 class), và tự tính 46,080 tham số. Buổi họp đối chiếu kết quả; ai lệch thì cả nhóm cùng tìm chỗ hiểu sai.
2. **Họp 2 lần/tuần (vd. T3 và T6), mỗi lần 45 phút:**
   - Teach-back: mỗi người 10 phút trình bày phần mình làm trong tuần; mỗi người nghe phải hỏi ≥1 câu.
   - Random viva 5 phút: bốc ngẫu nhiên một người giải thích một phần **không phải của mình**.
3. **Definition of Done của một PR:** code chạy + có test + reviewer giải thích lại được bằng lời ở buổi họp gần nhất.
4. **`docs/viva_questions.md`:** mỗi người thêm ≥3 câu hỏi/tuần và tự trả lời. Sau 05/12 dùng bộ câu này để tổng duyệt.
5. **Rotation Tuần 5:** mỗi người chạy lại một run của người khác chỉ dựa vào README. Nếu không chạy được thì README chưa đạt.
