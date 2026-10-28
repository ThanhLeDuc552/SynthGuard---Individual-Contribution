# Báo cáo: Train 4 CTGAN trên UNSW-NB15 (oversampled)

Cập nhật 25/10/2026 · Dự án SynthGuard (COS40005)

## 1. Mục tiêu

Train 4 CTGAN trên 4 phiên bản dataset UNSW-NB15 (chỉ gồm các bản ghi tấn công, nhiều rare attack hơn bản gốc), sinh dữ liệu synthetic cho từng bản, rồi dùng để train IDS và chuyển cho nhóm cyber chuẩn bị MIA. Model không cần hoàn hảo.

## 2. Dữ liệu đầu vào

4 file trong `dataset/UNSW_NB15/oversampled/`, đều 49 cột. Cột `Label` là hằng số (=1), cột nhãn loại tấn công là `Attack`.

| Attack | original | ros_10k | smote_3k | smote_10k |
|---|---|---|---|---|
| Analysis | 981 | 10000 | 3000 | 10000 |
| Backdoor | 3106 | 10000 | 3106 | 10000 |
| DoS | 4300 | 10000 | 4300 | 10000 |
| Exploits | 32390 | 32390 | 32390 | 32390 |
| Fuzzers | 21587 | 21587 | 21587 | 21587 |
| Generic | 6087 | 10000 | 6087 | 10000 |
| Reconnaissance | 9533 | 10000 | 9533 | 10000 |
| Shellcode | 1350 | 10000 | 3000 | 10000 |
| Worms | 112 | 10000 | 3000 | 10000 |
| **Tổng dòng** | **79446** | **123977** | **86003** | **123977** |

Lưu ý: `ros_10k` nhân bản các bản ghi gốc (Worms 112 dòng được lặp khoảng 89 lần). `smote_3k` và `smote_10k` tạo bản ghi mới bằng nội suy.

## 3. Cấu hình train

- Thư viện `ctgan`, chạy trên CPU (không có CUDA).
- 10 epoch, batch size 500, seed 42, giống nhau cho cả 4 bản.
- Bỏ cột hằng số `Label` khi train, gán lại sau khi sinh.
- Giá trị null chỉ có ở bản `original`, 2 cột `SRC_TO_DST_SECOND_BYTES` và `DST_TO_SRC_SECOND_BYTES` (6022 dòng mỗi cột), điền bằng median.
- Cột rời rạc duy nhất: `Attack`.
- Mỗi bản sinh số dòng synthetic bằng đúng số dòng file train.
- Thời gian: `original` 11,2 phút, `ros_10k` 15,0 phút, `smote_3k` 11,2 phút, `smote_10k` 15,9 phút, tổng 53,3 phút. Tốc độ khoảng 27,7 giây mỗi epoch với `original`, 32,8 với `ros_10k`, 22,8 với `smote_3k`, 32,1 với `smote_10k`.

## 4. Output

Trong thư mục `outputs/`, với mỗi bản `<name>` thuộc `original`, `ros_10k`, `smote_3k`, `smote_10k`:

- `ctgan_unsw_nb15_<name>.pt`
- `synthetic_ctgan_unsw_nb15_<name>.csv`

## 5. Kết quả kiểm tra nhanh: phân bố nhãn real và synthetic (%)

| Attack | original real | original syn | ros_10k real | ros_10k syn | smote_3k real | smote_3k syn | smote_10k real | smote_10k syn |
|---|---|---|---|---|---|---|---|---|
| Analysis | 1.23 | 9.36 | 8.07 | 10.93 | 3.49 | 10.24 | 8.07 | 10.85 |
| Backdoor | 3.91 | 10.99 | 8.07 | 10.90 | 3.61 | 10.35 | 8.07 | 10.85 |
| DoS | 5.41 | 11.38 | 8.07 | 10.73 | 5.00 | 10.48 | 8.07 | 10.85 |
| Exploits | 40.77 | 14.11 | 26.13 | 12.23 | 37.66 | 13.12 | 26.13 | 12.21 |
| Fuzzers | 27.17 | 13.77 | 17.41 | 11.62 | 25.10 | 12.73 | 17.41 | 11.80 |
| Generic | 7.66 | 11.75 | 8.07 | 10.91 | 7.08 | 10.97 | 8.07 | 10.89 |
| Reconnaissance | 12.00 | 12.42 | 8.07 | 10.81 | 11.08 | 11.64 | 8.07 | 10.91 |
| Shellcode | 1.70 | 9.92 | 8.07 | 10.98 | 3.49 | 10.17 | 8.07 | 10.80 |
| Worms | 0.14 | 6.28 | 8.07 | 10.88 | 3.49 | 10.30 | 8.07 | 10.85 |

Số liệu tính trực tiếp từ 4 file synthetic: không có giá trị null, cột `Label` luôn bằng 1, số dòng và số cột khớp file train.

Nhận xét:

- Cả 9 loại tấn công đều xuất hiện trong dữ liệu synthetic, kể cả các lớp hiếm.
- Phân bố nhãn synthetic gần như phẳng (khoảng 10–11% mỗi lớp), không theo tỉ lệ của file train: lớp lớn (Exploits, Fuzzers) bị giảm, lớp hiếm bị tăng.
- Nhiều khả năng do chỉ train 10 epoch nên generator chưa bám chặt điều kiện nhãn (chưa kiểm chứng).
- Hệ quả: 4 file synthetic có phân bố nhãn tương tự nhau dù đầu vào khác nhau, nên khác biệt giữa các bản chủ yếu ở mức đặc trưng và mức ghi nhớ record.
- Có thể sinh lại theo từng lớp (conditional sampling) để khớp tỉ lệ file train mà không cần train lại, nếu cần.

## 6. Câu hỏi mở cần lead xác nhận

1. **Holdout:** 4 CTGAN hiện train trên 100% mỗi file. 4 file oversampled được tạo từ toàn bộ dữ liệu hay từ một train split? MIA cần non-member mà generator (và cả bước oversample SMOTE) chưa từng thấy.
2. **Định nghĩa member:** member là bản ghi dùng để train IDS, hay bản ghi trong tập train của generator?
3. **Bản ghi SMOTE:** bản ghi SMOTE không phải record thật. Đề xuất gắn cờ bản ghi thật bằng cách khớp chính xác với file original, và chỉ tính member trên các bản ghi thật.
4. **ROS:** bản ghi bị nhân bản nhiều lần (Worms) nhiều khả năng bị ghi nhớ mạnh, cần ghi nhận khi đánh giá MIA.
5. **Bản synthetic gửi nhóm cyber:** dùng bản hiện tại (phân bố nhãn phẳng) hay bản sinh lại khớp tỉ lệ file train?

## 7. Bước tiếp theo

- Chốt các câu hỏi ở mục 6 với lead.
- Kiểm tra chất lượng (notebook 04) và train IDS baseline (notebook 05) trên cả 4 bản.
- Đóng gói gửi nhóm cyber: 4 file train, 4 file synthetic, danh sách bản ghi member/non-member sau khi chốt định nghĩa.
