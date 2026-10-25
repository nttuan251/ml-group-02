# ml-group-02 — Đồ án Intro to ML (HCMUS): tái lập & phản biện L2P

## Bối cảnh
- Đồ án cả học kỳ, làm theo nhóm, về một paper duy nhất: **L2P — Learning to Prompt for Continual Learning (Wang et al., CVPR 2022)**, file `Research/07_Improvement_L2P_Wang_CVPR2022.pdf`.
- Đề bài: https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/ (các tab reading / reproduction / analysis / empirics). Bản tóm tắt nằm trong `ROADMAP.md`.
- Sản phẩm nộp: (1) deck pptx 20–40 slide cho Reading + Reproduction (45%); (2) poster pptx A2 một slide cho Analysis + Empirics (50%); (3) chất lượng repo + W&B (5%). Cuối kỳ vấn đáp tại poster.

## Luật bắt buộc (vi phạm → 0 điểm)
- **Core của L2P phải tự cài đặt từ mô tả trong paper.** Code gốc (`google-research/l2p`, các bản PyTorch cộng đồng) chỉ được đọc/chạy để đối chiếu, **không copy** vào repo. Được dùng thư viện phụ (timm, torchvision, datasets), nhưng phải ghi trong README.
- Không chép text/hình của paper vào deck/poster.
- Mọi số liệu báo cáo phải gắn với một W&B run public. Seed cố định và được log.

## Môi trường
- **Deadline (Moodle):** link repo public T7 07/11/2026 23:59 · deck `ID.pptx` T7 21/11 23:59 · poster `ID.pptx` T7 05/12 23:59 (ID = số nhóm trên sheet lập nhóm). Freeze nội bộ vào thứ Năm trước mỗi hạn.
- Nhóm **3 người**, chia theo lát dọc + review chéo: A Method (prompt pool / selection / loss), B Data & Protocol (split, metric, Gaussian schedule, README, quota), C Training & Experiments (loop, logging, R1 headline, poster). Xem ROADMAP §3.
- Compute: chỉ có Kaggle T4 x2 (16GB mỗi GPU, không có bf16 → dùng fp16 AMP), **3 tài khoản, mỗi người một tài khoản** (không dùng tài khoản phụ — vi phạm điều khoản Kaggle). Mỗi session luôn chạy 2 run song song. Không có GPU local.
- Scope reproduction **Core+**: R1 Split CIFAR-100 (3 seed), R2 ablation, R3 Gaussian CIFAR-100, R4 sweep λ/M; 5-datasets là stretch. Logging "instrument once" để các run tái lập sinh luôn dữ liệu cho empirics.
- Python ≥3.9, PyTorch + timm. Dataset tải tự động từ HF/torchvision.

## Tài liệu trong repo
- `ROADMAP.md` — timeline theo tuần, mốc nộp, phân vai, ngân sách compute, rủi ro.
- `docs/l2p_spec.md` — đặc tả cài đặt, siêu tham số, số mục tiêu, các chỗ mơ hồ cần kiểm tra.
- `docs/experiments.md` — danh sách thí nghiệm R* (reproduction) và E* (empirics).
- `docs/critique.md` — ứng viên điểm mạnh / hạn chế / hướng phát triển.
- `docs/paper_selection.md` — lý do chọn L2P, các phương án dự phòng.
- `docs/kaggle_setup.md` — cách chạy trên Kaggle.
- `docs/viva_questions.md` — ngân hàng câu hỏi vấn đáp.

## Quy ước code
- Package `src/`, config YAML trong `configs/`, notebook Kaggle chỉ là lớp vỏ gọi `python -m src.train ...`.
- Không hardcode siêu tham số; log toàn bộ config lên W&B.
- Người dùng là sinh viên đang học: giải thích từ nền tảng, trả lời bằng tiếng Việt, giữ nguyên thuật ngữ kỹ thuật.
