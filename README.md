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

## 4. Trích dẫn tái sử dụng

## 5. Thông tin về datasets

## 6. Random seeds

## 7. Bảng ánh xạ thực nghiệm
