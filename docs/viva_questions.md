# Ngân hàng câu hỏi vấn đáp

Mỗi người thêm ≥3 câu/tuần **về phần không phải của mình**, rồi người sở hữu phần đó viết câu trả lời. Dùng cho random viva trong các buổi họp, và tổng duyệt sau 05/12.
Format: `- [chủ đề] Câu hỏi — (người hỏi → người trả lời)`, câu trả lời viết ngay bên dưới, thụt lề.

## Phương pháp
- [method] Vì sao query function q(x) phải là hàm cố định, không có tham số học? Nếu cho nó học thì chuyện gì xảy ra?
- [method] Top-N không khả vi. Vậy key học bằng cách nào? Key không được chọn có bao giờ được cập nhật không?
- [method] Vì sao lấy trung bình output tại các vị trí prompt mà không dùng token [CLS]?
- [method] Tự tính số tham số thêm vào với M=10, N=5, Lp=5, D=768. Vì sao chỉ khoảng 0.05%?
- [method] Diversified selection (Eq. 4) cần thông tin gì? Điều đó có mâu thuẫn với claim "task-agnostic" không?
- [complexity] Chi phí inference của L2P so với một ViT thường? Độ dài chuỗi tăng bao nhiêu và attention tăng theo bậc mấy?

## Protocol & metric
- [protocol] Class-incremental khác task-incremental và domain-incremental ở điểm nào? L2P được đánh giá ở setting nào?
- [metric] Viết công thức Forgetting. Vì sao lấy max theo thời gian mà không lấy accuracy ngay sau khi học xong task?
- [protocol] Gaussian-scheduled CIFAR-100 khác Split CIFAR-100 thế nào? Vì sao gọi là task-agnostic?

## Tái lập
- [repro] Vì sao FT-seq-frozen chỉ đạt khoảng 18% dù backbone đóng băng và feature rất tốt?
- [repro] Số của nhóm lệch bao nhiêu so với paper? Liệt kê từng nguồn sai lệch, và run nào đã kiểm chứng nguồn đó.
- [repro] Vì sao chọn Core scope? Bỏ những gì, và điều đó ảnh hưởng thế nào tới kết luận?
- [repro] Vì sao cần 3 seed? Std của nhóm so với std của paper thế nào?

## Phản biện
- [critique] Nếu baseline NCM trên feature đóng băng gần bằng L2P, điều đó nói gì về đóng góp thật của prompt?
- [critique] Vì sao chọn 2 điểm yếu này cho poster mà không chọn điểm thứ 3?
- [critique] Thí nghiệm nào của nhóm có thể đã bác bỏ giả thuyết của chính nhóm? Kết quả ra sao?
- [critique] Điểm yếu nào đã được paper khác chỉ ra? Đóng góp riêng của nhóm ở đó là gì?
