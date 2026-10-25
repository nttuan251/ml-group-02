# Ghi chép chọn paper (25/10/2026)

## Kết quả
- **Chọn: L2P** (CVPR 2022, A*). Nội dung form đăng ký: title "Learning to Prompt for Continual Learning"; year 2022; month 6; venue "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"; official link https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html; open-access để trống; quality A*; proof https://portal.core.edu.au/conf-ranks/604/
- **Dự phòng:** (1) Loss of Plasticity in Deep Continual Learning (Nature 2024); (2) CoTTA (CVPR 2022).

## Ràng buộc chọn bài (đề + sheet)
- Không quá 60 tháng kể từ ngày công bố; đúng chủ đề never-ending learning (continual / online / lifelong).
- Sheet có cột "Required citations" = 2 × số tháng kể từ ngày công bố. Venue được đánh giá theo CORE (conference) hoặc WoS (journal).
- Thầy reject các bài lạc chủ đề (HPO, data valuation, anomaly detection).

## So sánh ứng viên (citation từ Semantic Scholar, 10/2026)

| Bài | Citation | Ghi chú |
|---|---|---|
| L2P (CVPR'22) | 1474 | Gốc của nhánh prompt-based CL, chạy được trên T4 |
| DualPrompt (ECCV'22) | 989 | Follow-up của L2P |
| CoTTA (CVPR'22) | 871 | Gốc của continual TTA, rẻ, lý thuyết mỏng |
| EATA (ICML'22) | 762 | TTA |
| CODA-Prompt (CVPR'23) | 684 | Follow-up của L2P |
| SAR (ICLR'23) | 642 | TTA |
| RanPAC (NeurIPS'23) | 264 | **Đã bị Team 21 lấy** |
| Loss of Plasticity (Nature'24) | ~225 (gộp 2 bản) | Mới, hot, nhưng run tuần tự dài |
| ACIL (NeurIPS'22) | 114 | **Đã bị Team 20 lấy** |
| SimpleCIL/APER (IJCV'24) | 300 | **Đã bị Team 7 lấy** |
| Forget Forgetting (ICLR'26) | 12 | Quá mới; phần LLM không chạy nổi |

Bị loại vì quá 60 tháng: EWC, GEM, DER, GPM, TENT (#1, 2, 3, 4, 8 trong `Research/`), ER-ACE.
Đã có nhóm đăng ký (tính đến 24/10): RanPAC, ACIL, APER, FeCAM, cùng một số bài về online learning / concept drift.
