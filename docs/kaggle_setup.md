# Chạy trên Kaggle (T4 x2)

## Một lần cho mỗi tài khoản
0. **Mỗi thành viên chỉ dùng tài khoản của chính mình.** Điều khoản Kaggle cấm một người có nhiều tài khoản, và vi phạm có thể làm khóa cả tài khoản chính. Nhóm có 3 tài khoản, mỗi tài khoản đặt trần 24h/tuần.
1. Verify số điện thoại (Settings) để bật GPU + Internet cho notebook.
2. Add-ons → Secrets → thêm `WANDB_API_KEY` (lấy ở wandb.ai/authorize). **Không** dán key vào notebook hay commit lên repo.

## Mỗi notebook
- Accelerator: **GPU T4 x2**; Internet: **On**; Persistence: Files only (tùy chọn).
- Cell khởi động:

```python
from kaggle_secrets import UserSecretsClient
import os, subprocess
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
subprocess.run(["git", "clone", "https://github.com/<org>/ml-group-02.git"], check=True)
%cd ml-group-02
!pip install -q -r requirements.txt
```

- Chạy 2 seed song song, mỗi seed một GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/l2p_cifar100.yaml --seed 0 &
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/l2p_cifar100.yaml --seed 1 &
wait
```

## Lưu ý
- Session tối đa khoảng 12h; quota khoảng 30 GPU-giờ/tuần/tài khoản (kiểm tra lại trên trang Kaggle). Chạy dài thì dùng "Save & Run All (Commit)" để chạy nền.
- T4 không hỗ trợ bf16 → dùng `torch.autocast("cuda", dtype=torch.float16)` + `GradScaler`.
- Dataset tải tự động (torchvision/HF `uoft-cs/cifar100`); weight ViT tải từ HF Hub qua `timm`.
- Lưu checkpoint/feature cache vào `/kaggle/working`, rồi log cache thành W&B Artifact để tài khoản khác tái sử dụng.
- Gắn tag `account:<tên>` vào mỗi W&B run, để B tổng hợp quota đã dùng hằng tuần.
