# SynthGuard – AI workstream (CTGAN / TVAE / IDS baseline)

Pipeline cho nhóm AI: xem dataset → huấn luyện CTGAN (AI-1/2) và TVAE (AI-2) → sinh dữ liệu theo quota từng lớp (AI-3)
→ kiểm tra chất lượng (AI-3) → IDS baseline (AI-4). Vòng lặp lọc / sinh lại theo điểm rủi ro MIA (AI-5) cần điểm rủi ro của nhóm Cyber Security
(notebook 05 đã chừa chỗ: file `outputs/risk_scores_<ctgan|tvae>_<dataset>.csv` với một cột `risk_score`).

## Cài đặt (một lần)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
Trong VS Code: mở thư mục này, chọn Python interpreter là `.venv`, và cài extension **Python**, **Jupyter**, **Data Wrangler**.

## Đặt dữ liệu
```
data/unsw_nb15/   UNSW_NB15_training-set_cleaned.csv, UNSW_NB15_testing-set_cleaned.csv
data/cicids2017/  các file *.csv của CICIDS2017 (MachineLearningCSV)
```
Mỗi dataset được gọi riêng bằng biến `DATASET = "unsw_nb15"` hoặc `"cicids2017"` ở đầu notebook.

## Thứ tự chạy
1. `notebooks/01_view_datasets.ipynb` – xem bảng, số bản ghi từng lớp, các lớp hiếm.
2. `notebooks/02_train_ctgan.ipynb` – lần đầu để `SMOKE_TEST = True` như mặc định (vài phút), sau đó đặt `False` để huấn luyện thật.
   Đặt `LOAD_SAVED_MODEL = True` để nạp mô hình đã lưu thay vì huấn luyện lại.
3. `notebooks/03_train_tvae.ipynb` – như trên cho TVAE (dùng cùng CAP_PER_CLASS và QUOTA với notebook 02 để so sánh công bằng).
4. `notebooks/04_quality_check.ipynb` – hợp lệ, trùng lặp, khoảng cách tới bản ghi huấn luyện (DCR), số bản ghi từng lớp.
5. `notebooks/05_ids_baseline.ipynb` – Random Forest trên dữ liệu thật và thật + dữ liệu sinh; recall từng lớp hiếm.

Kết quả lưu ở `outputs/` (mô hình `.pt` và dữ liệu sinh `.csv`).

## Cấu trúc
```
src/config.py          đường dẫn, tên cột nhãn, cột bỏ, ngưỡng lớp hiếm
src/data.py            đọc dữ liệu, chuẩn bị cho CTGAN
src/ctgan_pipeline.py  huấn luyện CTGAN / TVAE, sinh theo quota (có lọc nhãn), kiểm tra nhanh
src/evaluation.py      hợp lệ, trùng lặp, DCR, hậu xử lý
src/ids.py             Random Forest baseline và so sánh thí nghiệm
notebooks/             các notebook 01–05
```

## Lưu ý
- CPU huấn luyện chậm. Mặc định `CAP_PER_CLASS` giảm các lớp lớn xuống còn 3000 dòng; lớp hiếm giữ nguyên.
- Không đưa dataset và `outputs/` lên Git (đã có trong `.gitignore`).
- Fidelity chính thức, chia train/holdout và MIA thuộc nhóm Data Science / Cyber Security; kiểm tra nhanh trong notebook chỉ là sơ bộ.
- Lớp hiếm: CTGAN chỉ dùng điều kiện lớp như gợi ý nên hàm sinh lọc lại nhãn và in `match_rate`. Lớp không đủ quota sẽ có cảnh báo, không bị độn thêm.
- Dữ liệu sinh có thể chứa giá trị không thể có (số âm). Notebook 04 đo tỷ lệ này; `POSTPROCESS` trong notebook 05 kéo về khoảng thật.
- Dữ liệu tham chiếu "thật chưa dùng huấn luyện" hiện là file testing của UNSW-NB15, chưa phải holdout chính thức (DS2).
