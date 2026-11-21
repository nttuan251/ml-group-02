# Learning to Prompt for Continual Learning - Paper Reading & Experimental Reproduction

**Giảng viên:**

- PGS.TS. Lê Hoàng Thái
- ThS. Huỳnh Lâm Hải Đăng

**Nhóm thực hiện:** Nhóm 2

**Thành viên:**

- 24122016 - Quách Lê Nhật Huy
- 24122024 - Trần Trọng Tú
- 24122025 - Nguyễn Thanh Tuấn

**Weights & Biases Project:** [Link](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT)

---

## 1. Giới thiệu đồ án

- **Tên và tác giả bài báo:** _Learning to Prompt for Continual Learning_ — Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, Tomas Pfister (CVPR 2022).
- **Link bài báo:** [https://arxiv.org/abs/2112.08654](https://arxiv.org/abs/2112.08654) (Official Repo: [google-research/l2p](https://github.com/google-research/l2p))
- **Reproduction scope:** Core Scope
- **Thông tin reproduction:** Tái hiện các thực nghiệm cốt lõi của L2P trên thiết lập class-incremental (CIFAR-100) và domain-incremental (CORe50). Thay thế cho các thí nghiệm phụ bằng phân tích mở rộng siêu tham số kiểm tra hiệu suất của phương pháp.

## 2. Hướng dẫn cài đặt và chạy thực nghiệm

### Chạy trên môi trường Kaggle Notebooks

- **Notebook:** Tìm ở `reproduction_code/notebook/l2p-reproduction.ipynb`
- Trước khi chạy, chọn accelerator **GPU T4 x2** và add on secret với key là `WANDB_API_KEY` và value là API key lấy từ W&B.

### Chạy trên máy local

- **Python Version:** Python `3.12.10+`
- **Installation:**
  ```bash
    git clone https://github.com/nttuan251/ml-group-02.git .
    pip install -r requirements.txt
    wandb login
    python -m reproduction_code.main
  ```

### Cách chỉnh cấu hình và dataset

- Điều chỉnh các hyperparameters trong các file ở `reproduction_code/experiment/experiment_*.py`.
- Sử dụng hàm tương ứng cho biến `run_config` trong `reproduction_code/main.py` tùy thuộc vào dataset muốn chạy.
- Nếu chạy trên kaggle tìm các khung code có nội dung thích hợp ở phần **3.2. Run**.

## 3. Giải thích các thư viện sử dụng

Ngoài các thư viện standard (`numpy`, `torch`, ...), nhóm còn sử dụng các thư viện:

- **`timm`**
  - **Nguyên nhân:** Cung cấp backbone ViT-B/16 đã pre-train (`vit_base_patch16_224.orig_in21k`) và trọng số tương ứng. Backbone là thành phần phụ trợ (bị đóng băng hoàn toàn), không phải đóng góp cốt lõi của L2P. Phần cốt lõi (prompt pool, query-key matching, ghép prompt, AvgPool trên vị trí prompt, loss) nhóm tự cài đặt, chỉ gọi lại `patch_embed`, `cls_token`, `pos_embed`, `blocks` và `norm` của ViT.
- **`torchvision`**
  - **Nguyên nhân:** Cung cấp các phép biến đổi ảnh chuẩn (`Resize` về 224 × 224, `ToTensor` đưa giá trị về [0, 1]) cho pipeline tiền xử lý.
- **`datasets` (Hugging Face)**
  - **Nguyên nhân:** Tải dữ liệu theo cách lập trình từ Hugging Face Hub (CIFAR-100, CORe50, ...) ngay trong lúc chạy, không cần chuẩn bị dữ liệu thủ công; hỗ trợ lọc theo lớp/session và biến đổi lazy.
- **`Pillow` (`PIL`)**
  - **Nguyên nhân:** Xử lý ảnh đầu vào (chuyển về RGB, ảnh grayscale của một số dataset); là thư viện mà `datasets` và `torchvision` dùng để giải mã ảnh.
- **`wandb` (Weights & Biases)**
  - **Nguyên nhân:** Theo dõi mọi lần chạy (cấu hình, accuracy, forgetting, tần suất chọn prompt) trên một project công khai để slide trích dẫn được từng run.
- **`kaggle_secrets`**
  - **Nguyên nhân:** Module có sẵn trên Kaggle Notebooks, dùng để đọc `WANDB_API_KEY` từ Kaggle Secrets thay vì ghi trực tiếp khóa vào notebook.

## 4. Trích dẫn tái sử dụng

Nhóm cài đặt L2P từ mô tả trong bài báo. Code chính thức ([google-research/l2p](https://github.com/google-research/l2p)) chỉ được đọc và chạy để đối chiếu (xem mục 5.3 của slide), không có đoạn code nào được sao chép vào bài làm.

Các thành phần được tái sử dụng:

| Thành phần | Dùng ở đâu | Nguồn |
|---|---|---|
| ViT-B/16 và trọng số pre-train ImageNet-21k (`vit_base_patch16_224.orig_in21k`) | Backbone đóng băng, kiêm hàm query | [1], [2] |
| Tensor, autograd, `Adam`, AMP (`autocast`, `GradScaler`) | Huấn luyện | [3], [4] |
| Phép biến đổi ảnh `Resize`, `ToTensor` | Tiền xử lý | [3] |
| Hugging Face `datasets` | Tải và lọc dữ liệu | [5] |
| Weights & Biases | Theo dõi thực nghiệm | [6] |
| CIFAR-100, CORe50 | Dữ liệu đánh giá | [7], [8] |

**Tài liệu tham khảo**

1. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in _Proc. ICLR_, 2021.
2. R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019. [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
3. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in _Proc. NeurIPS_, 2019.
4. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv:1412.6980, 2014.
5. Q. Lhoest et al., "Datasets: A community library for natural language processing," in _Proc. EMNLP (System Demonstrations)_, 2021.
6. L. Biewald, "Experiment tracking with Weights & Biases," 2020. [https://www.wandb.com](https://www.wandb.com)
7. A. Krizhevsky and G. Hinton, "Learning multiple layers of features from tiny images," Tech. Rep., 2009.
8. V. Lomonaco and D. Maltoni, "CORe50: a new dataset and benchmark for continuous object recognition," in _Proc. Conference on Robot Learning (CoRL)_, 2017.

## 5. Thông tin về datasets

## 6. Random seeds

Giá trị seed được lưu ở trường `random_seed` trong cấu hình của mỗi run và được W&B ghi lại:

| Nhóm thực nghiệm | Seed |
|---|---|
| Hai thí nghiệm chính (Split CIFAR-100 và CORe50) | **42, 24, 12** (3 lần chạy, báo cáo trung bình ± độ lệch chuẩn như paper) |
| Phân tích mở rộng (sweep M, sweep Lp × N, tần suất chọn prompt) | **42** (1 lần chạy) |

Seed được áp dụng ở các điểm sau:

| Vị trí | Cách áp dụng |
|---|---|
| Toàn cục | `set_global_seed(42)` đặt seed cho `random`, `numpy`, `torch` và `torch.cuda` |
| Thứ tự lớp của Split CIFAR-100 | `random.Random(seed)` xáo trộn 100 nhãn trước khi chia thành 10 task |
| Thứ tự batch khi huấn luyện | `torch.Generator().manual_seed(seed)` truyền vào `DataLoader` |
| Chọn mẫu đưa vào rehearsal buffer | `numpy.random.default_rng(seed)` (chọn mẫu theo lớp) và `torch.Generator` (lấy mẫu từ buffer) |
| Gaussian-scheduled CIFAR-100 | `random.Random(seed)` lấy mẫu thời điểm xuất hiện của từng ảnh |
| Khởi tạo prompt, key, classifier | Dùng seed toàn cục ở trên |

Lưu ý: nhóm không bật chế độ deterministic của cuDNN và có dùng mixed precision (fp16), nên chạy lại cùng seed trên GPU khác có thể lệch nhẹ về kết quả.

## 7. Bảng ánh xạ thực nghiệm

### 7.1. Nhóm đã thực hiện những thí nghiệm nào

Nhóm chọn **Core Scope**: không chạy lại các phương pháp so sánh (EWC, LwF, ER, DER++, ...), số liệu của các phương pháp này lấy từ paper. Nhóm chỉ chạy L2P và L2P-R, gồm:

1. **Hai thí nghiệm chính của paper** (mỗi thí nghiệm chạy với 3 seed: 42, 24, 12):
   - **Class-incremental trên Split CIFAR-100**: L2P (không buffer) và L2P-R (buffer 10/lớp và 50/lớp), đo Average Accuracy và Forgetting (Table 1 của paper).
   - **Domain-incremental trên CORe50**: L2P (không buffer) và L2P-R (buffer 50/lớp), đo Test Accuracy (Table 3 của paper).
2. **Phân tích mở rộng thay cho các thí nghiệm còn lại** (chỉ chạy với seed 42, trên Split CIFAR-100, không buffer):
   - **Độ nhạy theo kích thước prompt pool M** (10, 15, 20, 25, 30) với Lp = 5, N = 5.
   - **Độ nhạy theo độ dài prompt Lp và số prompt chọn N** với M = 20 (dạng heatmap, đối chiếu với Figure 4 của paper).
   - **Tần suất chọn prompt** theo task và theo M, để xem mức chia sẻ prompt giữa các task (đối chiếu với Figure 3 của paper).

Các thí nghiệm không thực hiện: 5-datasets, Gaussian-scheduled CIFAR-100 (task-agnostic), ablation và các baseline.

### 7.2. Ánh xạ thí nghiệm, code và run W&B

Tất cả run nằm trong W&B project [`L2P_PROJECT`](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT). Mọi thí nghiệm được tạo từ notebook `reproduction_code/notebook/l2p-reproduction.ipynb`, thay đổi cấu hình bằng hàm `get_*_config` (hoặc thay các trường như `rehearsal_samples_per_class`, `prompt_pool_size`, `prompt_length`, `top_n_prompts`). Mỗi liên kết ở cột "W&B" là một report tổng hợp các run của thí nghiệm đó (thí nghiệm chính gồm 3 run ứng với 3 seed).

| Thí nghiệm | Cấu hình | W&B | Kết quả cho thấy |
|---|---|---|---|
| Split CIFAR-100, L2P (không buffer) | `get_CIFAR100_config`, `rehearsal_samples_per_class=0` | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Class-incremental-learning-Split-CIFAR-100--VmlldzoxODMzNDg3NA) | Average Acc và Forgetting của L2P so với paper (slide 26–27) |
| Split CIFAR-100, L2P-R (buffer 10/lớp) | `get_CIFAR100_config`, `rehearsal_samples_per_class=10` | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Class-incremental-learning-Split-CIFAR-100--VmlldzoxODMzNDg3NA) | Như trên, khi có buffer nhỏ |
| Split CIFAR-100, L2P-R (buffer 50/lớp) | `get_CIFAR100_config`, `rehearsal_samples_per_class=50` | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Class-incremental-learning-Split-CIFAR-100--VmlldzoxODMzNDg3NA) | Như trên, khi có buffer lớn |
| CORe50, L2P (không buffer) | `get_CORe50_config`, `rehearsal_samples_per_class=0` | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Domain-incremental-learning-CORe50--VmlldzoxODMzNTgxNA) | Test accuracy domain-incremental so với paper (slide 28) |
| CORe50, L2P-R (buffer 50/lớp) | `get_CORe50_config`, `rehearsal_samples_per_class=50` (thực tế 48/lớp, 6/lớp/task) | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Domain-incremental-learning-CORe50--VmlldzoxODMzNTgxNA) | Như trên, khi có buffer |
| Sweep kích thước pool M (10, 15, 20, 25, 30) | Split CIFAR-100, Lp=5, N=5, không buffer | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Prompt-Pool-Size-Sensitivity-Split-CIFAR-100--VmlldzoxODMzNTk3NQ) | Accuracy ổn định theo M (slide 32) |
| Sweep độ dài prompt Lp × số prompt chọn N | Split CIFAR-100, M=20, tag `HYPERPARAM_SWEEP`, tên run dạng `sweep_M20_Lp{Lp}_N{N}` | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Hyperparameter-HeatmapSplit-CIFAR-100--VmlldzoxODMzNTg5OQ) | Ảnh hưởng của Lp và N (slide 35) |
| Phân tích tần suất chọn prompt theo task và theo M | Dựng từ các run sweep M ở trên (`prompt_usage/*`) | [Report](https://forge.coreweave.com/wandb/nhuy165-org/L2P_PROJECT/reports/Prompt-Pool-Size-Sensitivity-Split-CIFAR-100--VmlldzoxODMzNTk3NQ) | Mức chia sẻ prompt giữa các task (slide 33–34) |
