# Ứng viên phản biện (Phase 3)

Đề bài chỉ phê bình **lý thuyết** ở giai đoạn này: giả định ẩn, tính đúng đắn của claim, hàm mục tiêu, độ phức tạp/tổng quát hóa. Mỗi hạn chế phải đủ cụ thể để thiết kế được một thí nghiệm (xem `experiments.md`).
Đây mới là **giả thuyết** để kiểm chứng; chưa có số liệu nào cả.

## Điểm mạnh (cần ≥3, mỗi điểm kèm bằng chứng)

1. **Rehearsal-free:** không lưu dữ liệu cũ (hợp với bài toán privacy), mà vẫn vượt các phương pháp rehearsal buffer nhỏ (Table 1). Bằng chứng: số tái lập R1a.
2. **Tham số thêm rất ít:** khoảng 46K, tức 0.05%. Có thể tự tính từ M·Lp·D + M·D.
3. **Không cần task id lúc test:** chọn prompt theo từng instance (Eq. 3). Bằng chứng: R3 (task-agnostic).
4. Tách quá trình học query khỏi học prompt nhờ key–value. Bằng chứng: ablation R2.

## Hạn chế lý thuyết (cần ≥2)

Owner (xem ROADMAP §3): **L1 → C**, **L2 → B**, **L3/L4 → A**, **L5 → A** (phụ). Giả thuyết phải được viết xong trước CN 15/11; chọn 2 hạn chế lên poster vào CN 29/11.

| # | Hạn chế | Lập luận | Thí nghiệm |
|---|---|---|---|
| L1 | **Kiến thức thực chất nằm ở backbone pretrain.** Query q(x) đóng băng, nên việc chọn prompt chỉ tốt khi feature pretrain đã tách được các task. ImageNet-21k chồng lấn ngữ nghĩa với CIFAR. Claim "học liên tục không quên" bị lẫn với "feature tốt sẵn có". | Giả định ẩn về phân phối | E1, E2 |
| L2 | **"Task-agnostic" không hoàn toàn.** Diversified selection dùng tần suất theo task (cần biết ranh giới task); head có thể mask logit theo class của task hiện tại; giao thức đánh giá vẫn chia task rõ ràng. | Claim vs giả định | E3, E7 |
| L3 | **Hàm mục tiêu của key giống k-means/VQ:** chỉ key được chọn mới nhận gradient → dễ có key "chết" và pool sụp về vài prompt dùng chung. Fig. 3 trái cho thấy các task CIFAR dùng chung gần hết prompt, tức là thực chất gần với một prompt chung, vốn là cấu hình mà chính ablation của paper nói là quên nhiều. | Hàm mục tiêu / lời giải suy biến | E4, R4 (sweep λ, M) |
| L4 | **Prompt dùng chung vẫn bị ghi đè:** prompt được cập nhật ở mọi task chọn nó, không có cơ chế bảo vệ, nên forgetting chuyển từ trọng số sang prompt. Head g dùng chung cũng bị task-recency bias. | Giả định ổn định | E5, E3 |
| L5 | **Chi phí:** inference cần 2 forward ViT; độ dài chuỗi tăng N·Lp nên attention tăng theo bậc hai. Claim "succinct memory" chỉ đúng về **bộ nhớ**, không đúng về compute. | Độ phức tạp | E6 |

## Hướng phát triển (cần ≥2, mỗi hướng gắn một hạn chế)

- **L1 →** đánh giá bằng backbone/dataset không chồng lấn; hoặc thêm một bước thích nghi nhẹ ở task đầu. Chỉ trích dẫn ý tưởng first-session adaptation, không dùng code của nhóm khác.
- **L3 →** khởi tạo key bằng k-means trên query của task đầu, hoặc thêm regularizer đa dạng/trực giao cho key (ý tưởng gần CODA-Prompt).
- **L4 →** freeze prompt hoặc head theo task, hoặc thay head bằng NCM theo class mean; ý tưởng classifier alignment như SLCA.
- **L5 →** lấy query từ layer nông hoặc cache query, để giảm từ 2 forward xuống khoảng 1.x.

## Trích dẫn khi điểm đã được biết

Một số điểm đã được các follow-up chỉ ra một phần: DualPrompt (prompt chung/riêng), CODA-Prompt (selection không khả vi, top-N), SLCA (head bị lệch). Khi dùng những điểm này, phải trích dẫn và nói rõ phần đóng góp riêng của nhóm (thí nghiệm mới, góc nhìn lý thuyết mới). Ưu tiên các điểm mang tính khám phá riêng: **E3** (class mask ngầm), **E4** (prompt collapse đo trực tiếp), **E7** (thứ tự task).

## Lưu ý tránh trùng nhóm khác

Các nhóm khác đã đăng ký RanPAC, ACIL, SimpleCIL/APER, FeCAM (cùng mảng PTM-based CIL). Nhóm mình chỉ dùng baseline **tự cài đặt** (NCM, linear probe) và trích dẫn như related work, không dựa vào sản phẩm của họ.
