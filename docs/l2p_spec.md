# L2P — Đặc tả cài đặt (tự viết từ paper)

Nguồn: Wang et al., CVPR 2022, `Research/07_Improvement_L2P_Wang_CVPR2022.pdf` (§3–4, §5.2, Appendix C–D).
File này là "hợp đồng" để tự cài đặt core. Chỗ nào paper không nói rõ thì ghi ở §6, đối chiếu với code gốc **chỉ bằng cách đọc**.

## 1. Bài toán

- Chuỗi task D = {D_1, …, D_T}, D_t = {(x_i^t, y_i^t)}. Khi học task t, không còn truy cập dữ liệu task < t (buffer = 0).
- **Class-incremental:** các task có tập class rời nhau; lúc test **không biết task id**, dự đoán trên toàn bộ class đã thấy.
- Mục tiêu: một model f dự đoán đúng trên mọi task đã học.

## 2. Thành phần

Ký hiệu: ViT đóng băng f = f_r ∘ f_e; f_e là patch embedding (cùng [CLS] và positional embedding), f_r là các block transformer. D = 768 với ViT-B/16; L = 197 token (196 patch + CLS).

| Thành phần | Dạng | Train? |
|---|---|---|
| Prompt pool P = {P_1..P_M} | mỗi P_j ∈ R^{Lp×D} | ✅ |
| Key K = {k_1..k_M} | mỗi k_j ∈ R^{D_k}, với D_k = D | ✅ |
| Query q(x) = f(x)[0,:] | feature [CLS] của ViT **không prompt** | ❌ (frozen) |
| Matching γ(q, k) | cosine distance = 1 − cos(q, k) | — |
| Head g | Linear(D → tổng số class) | ✅ |
| Backbone f_e, f_r | ViT-B/16 pretrained | ❌ |

## 3. Forward (một sample x)

1. q = f(x)[0,:] (một forward không grad).
2. Chọn top-N key nhỏ nhất theo γ(q, k_j) → chỉ số s_1..s_N (Eq. 3).
   - *Tùy chọn khi train, nếu biết ranh giới task:* dùng γ(q, k_j) · h_j, trong đó h_j là tần suất chuẩn hóa của prompt j được chọn **đến hết task t−1** (Eq. 4). Lúc test luôn dùng Eq. 3. Paper chỉ dùng tùy chọn này cho 5-datasets.
3. x_e = f_e(x) ∈ R^{L×D}; x_p = [P_{s1}; …; P_{sN}; x_e] ∈ R^{(N·Lp+L)×D} (Eq. 2).
4. h = f_r(x_p); z = AvgPool(h[0 : N·Lp, :]), tức là chỉ lấy trung bình output tại **các vị trí prompt**.
5. logits = g(z).

## 4. Loss & tối ưu

L = CE(g(z), y) + λ · Σ_{i∈K_x} γ(q(x), k_{s_i})   (Eq. 5)

- Số hạng thứ hai là surrogate loss kéo các key **được chọn** về phía query. Key không được chọn không nhận gradient.
- Theo Algorithm 1: chỉ cập nhật các (k, P) được chọn trong batch, cùng với head φ.
- Top-N không khả vi; gradient tới key chỉ đi qua surrogate loss.

## 5. Siêu tham số (paper §5.2)

| Tham số | Split CIFAR-100 | 5-datasets |
|---|---|---|
| M (pool) | 10 | 20 |
| N (chọn) | 5 | 4 |
| Lp | 5 | 5 |
| λ | 0.5 | 0.5 |
| Diversified selection | không | có |
| Optimizer | Adam (β1=0.9, β2=0.999), lr hằng 0.03 | như bên trái |
| Batch | 128 | 128 |
| Epoch/task | 5 | 5 |
| Input | resize 224×224, scale về [0,1] (không chuẩn hóa mean/std) | như bên trái |
| Backbone | ViT-B/16 pretrained (xem §6) | như bên trái |
| Số seed | 3 | 3 |

Số tham số thêm: M·Lp·D = 10·5·768 = 38,400 prompt + 10·768 = 7,680 key → 46,080 (khớp paper).

### Số mục tiêu (Table 1, buffer 0, Split CIFAR-100)

| Method | Avg Acc ↑ | Forgetting ↓ |
|---|---|---|
| FT-seq-frozen | 17.72 ± 0.34 | 59.09 ± 0.25 |
| FT-seq | 33.61 ± 0.85 | 86.87 ± 0.20 |
| EWC | 47.01 ± 0.29 | 33.27 ± 1.17 |
| LwF | 60.69 ± 0.63 | 27.77 ± 2.17 |
| **L2P** | **83.83 ± 0.04** | **7.63 ± 0.30** |
| Upper-bound | 90.85 ± 0.12 | — |

5-datasets: L2P 81.14 ± 0.93 / 4.64 ± 0.52. Ablation (Table 5, 5-datasets): w/o pool 51.96/26.60; w/o key–value 58.33/20.45; w/o diversified 62.26/17.84.

### Metric
- a_{t,i}: accuracy trên test set của task i sau khi học xong task t.
- **Average accuracy** = (1/T) Σ_i a_{T,i}.
- **Forgetting** = (1/(T−1)) Σ_{i<T} max_{t∈[i,T−1]} (a_{t,i} − a_{T,i}).
- Kiểm tra lại định nghĩa chính xác trong các ref [7, 32, 34] của paper trước khi chốt, và ghi rõ trong deck.

## 6. Chỗ mơ hồ — cần kiểm tra bằng cách *đọc* code gốc

Mỗi điểm dưới đây vừa là nguồn sai lệch khi tái lập, vừa có thể là ý cho phần critique. Hãy ghi kết quả kiểm tra vào đây.

1. **Checkpoint backbone:** paper chỉ ghi "pre-trained ViT-B/16". Cần xác định là ImageNet-21k thuần hay 21k → fine-tune 1k (bản `vit_base_patch16_224` của timm là augreg in21k-ft-in1k). Điểm này ảnh hưởng lớn tới số và tới critique về chồng lấn dữ liệu pretrain.
2. **Class mask ở head:** paper không nhắc. Khi train task t, logits của class ngoài task t có bị mask không? Nếu có thì đây là thông tin task ngầm, mâu thuẫn với tinh thần "task-agnostic".
3. **Augmentation khi train:** paper không nêu. Điều này quyết định có cache được q(x) hay không.
4. **Thứ tự class** của Split CIFAR-100: cố định hay shuffle theo seed?
5. **Head:** một head chung dùng cho mọi task hay mở rộng dần? Có reset/khởi tạo lại không?
6. **Surrogate loss** lấy trung bình hay tổng theo batch và theo N?
7. **Pool được khởi tạo** thế nào (uniform? normal?) và key có được chuẩn hóa không?
8. **Loss có trừ hay cộng** số hạng matching (cosine similarity với cosine distance)?
9. **Batch 128 với T4 16GB:** khi dùng fp16 có thể vừa; nếu không thì dùng grad accumulation và ghi rõ trong deck.

## 6b. Deviation log — chỗ nhóm cố ý làm khác paper

Theo ROADMAP §4 quy tắc 7: bản `faithful` giữ đúng paper, bản `improved` bật bằng config flag. Mỗi thay đổi một dòng, ghi ngay khi quyết định, không để dồn.

| # | Thành phần | Paper làm | Vấn đề nhóm thấy (lý thuyết hoặc số liệu) | Nhóm thay bằng | Config flag | Run so sánh (W&B) | Người đề xuất / review |
|---|---|---|---|---|---|---|---|
| D1 | | | | | | | |

## 7. Độ phức tạp (cho slide phương pháp + critique)

- **Train:** mỗi sample cần 1 forward không grad (query) + 1 forward/backward với độ dài L + N·Lp = 197 + 25 = 222 token. Attention có chi phí O((L+N·Lp)²·D) cho mỗi layer.
- **Inference:** **2 lần forward ViT** cho mỗi ảnh, tức gấp khoảng 2× fine-tune thường.
- **Bộ nhớ thêm:** O(M·Lp·D + M·D), không đáng kể. Nhưng backward vẫn phải lưu activation của toàn bộ ViT, vì prompt nằm ở input.
- **So sánh:** ER cần bộ nhớ O(buffer × kích thước ảnh); EWC cần O(|θ|) cho Fisher.
