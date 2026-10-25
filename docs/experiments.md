# Kế hoạch thí nghiệm

Mọi run phải log lên W&B public, gắn `seed`, `config`, tag theo giai đoạn (`lab:repro` hoặc `lab:empirics`) và `account:<tên>`. Mỗi ID dưới đây ↔ một dòng trong bảng ánh xạ của README.

## Reproduction — scope Core+ (deadline deck 21/11)

**Lý do chọn Core:** chỉ có 4 tuần cho 3 người. Đề yêu cầu tối thiểu 2 thí nghiệm quan trọng nhất (bao gồm headline) cộng một phân tích tự thiết kế. R1 + R3 + R4 đáp ứng đủ, R2 là phần thêm. GPU không phải giới hạn (xem ROADMAP §5).

| ID | Thí nghiệm | So với paper | Seed | Owner | GPU-h/run (ước tính) | Xong trước |
|---|---|---|---|---|---|---|
| R1a | L2P, Split CIFAR-100, buffer 0 | Table 1: 83.83 ± 0.04 / 7.63 ± 0.30 | 3 | C | 1–1.5 | 15/11 |
| R1b | FT-seq-frozen | Table 1: 17.72 ± 0.34 / 59.09 ± 0.25 | 3 | C | <0.2 (feature cache) | 15/11 |
| R1c | Upper-bound (i.i.d.) | Table 1: 90.85 ± 0.12 | 1–3 | C | 1–2 | 15/11 |
| R2 | Ablation trên Split CIFAR-100: w/o prompt pool (1 prompt), w/o key–value (key = mean prompt) | Table 5 — chỉ so **xu hướng**, vì paper làm trên 5-datasets | 1–3 | A | 1–1.5 | 15/11 |
| R3 | Gaussian-scheduled CIFAR-100 (task-agnostic, mỗi batch train 1 lần) | Table 4: 88.34 ± 0.14 | 3 | B | ~0.5 | 15/11 |
| R4 | **Tự thiết kế:** sweep λ ∈ {0, 0.1, 0.5, 1, 2} và M ∈ {5, 10, 20} | Paper khẳng định λ "không nhạy"; Fig. 4 phải | 1 | A | 1–1.5 | 19/11 |
| R5 (stretch) | 5-datasets (M=20, N=4, có diversified selection) | Table 1: 81.14 ± 0.93 / 4.64 ± 0.52 | 1 | B | 4–6 | chỉ làm nếu G3 đạt sớm |

Ghi chú:
- Sai lệch nào cũng phải được giải thích, tốt nhất bằng một run kiểm chứng (xem `l2p_spec.md` §6).
- R5 cần notMNIST, vốn không có trên torchvision. Nếu làm, phải ghi rõ nguồn và kích thước bản dùng.

## Instrument once — log ngay từ Tuần 2 (owner: C, review: A)

Chỉ có 2 tuần giữa deck và poster, nên mọi run tái lập phải log sẵn các đại lượng dưới đây. Khi đó phần lớn empirics chỉ là **phân tích lại log**, không phải chạy mới.

| Đại lượng | Khi nào log | Phục vụ |
|---|---|---|
| Accuracy matrix a_{t,i} (T×T) | sau mỗi task | Avg Acc, Forgetting, E5 drift |
| Histogram chọn prompt theo task (train và test) | sau mỗi task | E4 collapse, prompt chết, độ chính xác chọn prompt |
| Số prompt chưa từng được chọn; entropy của phân phối chọn | sau mỗi task | E4, L3 |
| Khoảng cách cosine trung bình query–key | mỗi epoch | E4 |
| L2 norm của trọng số head theo nhóm class của từng task | sau mỗi task | E3 task-recency bias |
| Confusion matrix theo task (đoán nhầm sang task nào) | cuối run | E3, L2 |
| Throughput (ảnh/giây) và bộ nhớ GPU đỉnh | đầu run | E6, ngân sách compute |
| Snapshot prompt + key sau mỗi task (file nhỏ, ~46K số) | sau mỗi task | E5 đo prompt drift |

## Empirics (deadline poster 05/12) — chạy Tuần 5, giả thuyết viết từ Tuần 3

Ánh xạ tới các hạn chế trong `critique.md`.

| ID | Kiểm chứng | Loại | Owner | Chi phí |
|---|---|---|---|---|
| E1 | Prompt có thật sự cần? So L2P với NCM / linear probe tự cài trên feature [CLS] đóng băng | Baseline thay thế | C | Rất rẻ (feature cache) — làm sớm từ Tuần 3 |
| E2 | Phụ thuộc pretrain: đổi backbone (in1k-only, self-supervised DINO) và/hoặc dataset xa ImageNet | Vi phạm giả định / OOD | C | Đắt nhất — chạy ngay đầu Tuần 5 |
| E3 | Bỏ class mask ở head; đo bias của head về task mới (norm, confusion) | Ablation | B | Trung bình |
| E4 | Prompt collapse: entropy chọn, số prompt chết, độ chính xác chọn prompt khi test | Phân tích cơ chế | A | Gần như miễn phí (từ log) |
| E5 | Prompt drift: thay đổi của prompt dùng chung qua các task; so với biến thể freeze prompt | Ablation | A | Trung bình |
| E6 | Latency và bộ nhớ theo N·Lp, so với 1 forward thường | Scaling | A | Rẻ |
| E7 | Thứ tự task / độ giống nhau giữa các task (xếp task theo superclass CIFAR-100) | Vi phạm giả định | B | Trung bình |

## Quy ước

- Seed mặc định: 0, 1, 2. Class order sinh từ seed và lưu kèm run.
- Config ở `configs/*.yaml`. Không hardcode siêu tham số trong code.
