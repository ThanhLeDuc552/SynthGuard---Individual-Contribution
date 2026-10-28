Ý tưởng chung: muốn biết một bộ data synthetic tốt hay không, cần hỏi 5 câu. Mỗi câu là một nhóm metric:

| Nhóm | Câu hỏi |
|---|---|
| **Coverage** | Có sinh **đủ** các loại attack không? |
| **Validity** | Từng dòng có **hợp lệ** không, hay chứa giá trị vô lý? |
| **Fidelity** | Phân phối có **giống** data thật không? |
| **Privacy** | Có **copy hoặc học thuộc** data thật không? |
| **Utility** | Dùng để train thì có **giúp model** không? |

Một bộ data tốt phải đạt cả 5. Ví dụ, ROS rất "giống thật" (Fidelity cao) nhưng đó là vì nó copy nguyên dòng (Privacy kém).

---

## Các thành phần chung

| Code | Vai trò |
|---|---|
| [`Preprocessor`](ml_ids_detect.py#L39) | Đổi `NaN`/`inf` thành 0. Có thể kẹp giá trị về khoảng [min, max] của train thật. |
| [`group_split`](ml_ids_detect.py#L76) | Chia train/test sao cho các dòng giống hệt nhau nằm cùng một phía. |
| [`make_xgb`](ml_ids_detect.py#L82) | Tạo model XGBoost (CPU hoặc GPU). |
| [`new_rows`](ml_ids_detect.py#L192) | Chỉ giữ phần dòng **sinh thêm**: trừ đi các dòng data thật có sẵn trong file, và bỏ các dòng trùng test. |
| [`compare`](ml_ids_detect.py#L208) | Hàm chính: train mốc ([dòng 218](ml_ids_detect.py#L218)), lặp qua từng file ([dòng 241](ml_ids_detect.py#L241)), ghi kết quả ([dòng 314](ml_ids_detect.py#L314)). |

---

## 1. Coverage: có sinh đủ không ([dòng 259–262](ml_ids_detect.py#L259-L262))

Đếm số loại attack có trong file synthetic, so với 9 loại của data thật.

- `classes_covered`: ví dụ `5/9`.
- `missing_classes`: các loại bị thiếu.

Nếu generator chỉ sinh ra các loại phổ biến và bỏ qua loại hiếm, hiện tượng đó gọi là **mode collapse**. Ví dụ, TVAE bản `original` chỉ sinh được 5/9 loại; mất Backdoor, DoS, Shellcode, Worms. Đây lại chính là các loại cần sinh thêm nhất.

## 2. Validity: có hợp lệ không ([dòng 264–269](ml_ids_detect.py#L264-L269))

Trước vòng lặp, code học 3 điều từ data thật:
- Cột nào **không bao giờ âm** (bytes, packets, duration...).
- Cột nào **luôn là số nguyên** (cột đếm).
- **Giá trị lớn nhất** của từng cột.

Sau đó kiểm tra từng ô synthetic:

| Cột | Ý nghĩa |
|---|---|
| `neg_cell_rate` | Tỉ lệ ô bị âm ở cột không bao giờ âm. Ví dụ `IN_BYTES = -350` là không thể có. |
| `nonint_cell_rate` | Tỉ lệ ô là số lẻ ở cột đếm. Ví dụ `IN_PKTS = 3.7` gói là vô lý. |
| `over_max_cell_rate` | Tỉ lệ ô vượt max của data thật. |
| `invalid_row_rate` | Tỉ lệ dòng có **ít nhất một** ô vô lý. |

Kết quả của bạn: CTGAN có khoảng 17% ô bị âm, và gần 100% dòng có ít nhất một lỗi. Một flow như vậy không thể xuất hiện trên mạng thật.

## 3. Fidelity: có giống thật không ([dòng 271–285](ml_ids_detect.py#L271-L285))

Điểm quan trọng: code so **từng loại attack với chính loại đó** trong data thật (synthetic Worms với real Worms, synthetic DoS với real DoS). Nếu so gộp tất cả, file oversample sẽ bị phạt chỉ vì nó cố ý đổi tỉ lệ giữa các lớp.

| Metric | Cách đo | Ví dụ |
|---|---|---|
| `ks_mean` | **KS test**: lấy khoảng cách lớn nhất giữa hai đường phân phối tích lũy (CDF) của real và syn, tính trên từng cột rồi lấy trung bình. 0 là giống hệt, 1 là khác hoàn toàn. | Nếu real `IN_BYTES` phần lớn khoảng 200 còn syn khoảng 5000, KS sẽ cao. |
| `ks_worst_feature` | Cột lệch nhiều nhất theo KS. | Gợi ý cho người train generator nên sửa cột nào. |
| `corr_diff_mean` | So **mối quan hệ giữa các cột** (tương quan Spearman). | Thật: `IN_PKTS` tăng thì `IN_BYTES` tăng. Nếu syn phá vỡ quan hệ này, dù từng cột riêng lẻ trông giống thật thì từng dòng vẫn vô lý. |
| `tail_q99_log_ratio` | So **phần đuôi** của phân phối (phân vị 99%). | Các flow cực lớn (burst, DoS) thường bị generator "làm mượt" mất. |
| `L1_pktsize_entropy_diff` | Phân bố kích thước gói (`NUM_PKTS_*`) có đa dạng giống thật không. | |
| `L2_direction_jsd` | Tỉ lệ bytes chiều đi so với chiều về. | Flood thường chỉ một chiều, phiên tương tác thì hai chiều. |
| `L3_flag_*` | TCP flags có giống thật không, và có phải tổ hợp flag hợp lệ không. | |

L1–L3 là các metric riêng cho dữ liệu mạng, lấy từ sheet. Code của chúng nằm ở [`network_metrics`](ml_ids_detect.py#L163).

## 4. Privacy: có copy data thật không ([dòng 287–293](ml_ids_detect.py#L287-L293))

**DCR** (Distance to Closest Record): với mỗi dòng synthetic, tìm dòng thật gần nó nhất và đo khoảng cách. Trước khi đo, dữ liệu được đưa về thang log rồi chuẩn hóa, để cột bytes (hàng triệu) không lấn át cột flags (vài chục).

| Cột | Ý nghĩa |
|---|---|
| `exact_copy_rate` | Khoảng cách bằng 0, tức là **copy nguyên dòng**. ROS ra 65%. |
| `dcr_train_p5` / `dcr_holdout_p5` | 5% dòng synthetic gần data thật nhất thì gần đến mức nào. |
| `closer_to_train_rate` | Tỉ lệ dòng synthetic **gần train hơn gần holdout**. |

Ý tưởng của `closer_to_train_rate`: generator chỉ được thấy train, không thấy holdout (test). Nếu nó **học quy luật chung**, dòng sinh ra sẽ gần train và holdout như nhau, cho kết quả khoảng 0.5. Nếu nó **học thuộc** train, dòng sinh ra sẽ gần train hơn hẳn, cho kết quả lớn hơn nhiều so với 0.5. Code lấy mẫu train bằng đúng cỡ holdout ([dòng 236](ml_ids_detect.py#L236)) để so công bằng.

Lưu ý: khoảng cách lớn chưa chắc là tốt. CTGAN có DCR khoảng 16, rất xa mọi dòng thật. Đó không phải vì nó bảo vệ privacy tốt, mà vì data của nó không giống thật.

## 5. Utility: có giúp model không ([dòng 295–310](ml_ids_detect.py#L295-L310))

Đây là nhóm quan trọng nhất. Tất cả model đều được **test trên data thật**; chỉ khác nhau ở data dùng để train:

```
TRTR  = Train Real            → Test Real     (mốc so sánh)
TSTR  = Train Synthetic       → Test Real     (synthetic có thay thế được data thật không?)
TR+S  = Train Real + Synthetic → Test Real    (thêm synthetic có giúp được không?)
```

Hàm [`train_and_score`](ml_ids_detect.py#L141) train XGBoost phân loại loại attack, rồi trả về macro-F1, MCC, F1 từng lớp và độ quan trọng của từng feature (SHAP).

| Cột | Ý nghĩa |
|---|---|
| `tstr_macro_f1`, `tstr_mcc` | Điểm khi train **chỉ bằng synthetic**. Cao nghĩa là synthetic chứa đủ thông tin để học. |
| `ml_efficacy_ratio` | TSTR / TRTR. Bằng 1 nghĩa là synthetic tốt ngang data thật. TVAE đạt khoảng 0.58, CTGAN khoảng 0.13–0.25. |
| `trs_macro_f1` | Điểm khi **trộn** synthetic vào data thật. Đây là cách dùng thực tế (giống `ids_netflow.py`). |
| `aug_gain_macro_f1` | TR+S − TRTR. **Dương nghĩa là synthetic giúp, âm nghĩa là làm hại.** |
| `aug_gain_rare_f1` | Như trên nhưng chỉ tính các lớp hiếm (Worms, Shellcode, Analysis), là mục đích chính của việc sinh data. |
| `shap_rank_stability` | Model train bằng synthetic có dựa vào **các feature giống** model train bằng data thật không. Thấp nghĩa là model học được các "mẹo" sai từ synthetic. |

**Vì sao cần cả TSTR lẫn TR+S?**
- **TSTR** đo chất lượng của **riêng** synthetic. Nó khắt khe, nhưng tách bạch được generator tốt và kém.
- **TR+S** đo **lợi ích thực tế** khi dùng. Một bộ synthetic có TSTR thấp vẫn có thể giúp TR+S nếu nó bổ sung đúng các lớp hiếm. Ngược lại, TSTR cao cũng chưa chắc giúp được nếu data thật đã đủ.

Với data của bạn, CTGAN và TVAE đều có `aug_gain` hơi âm, chỉ SMOTE 10k là dương (+0.03).

---

## Đầu ra ([dòng 314](ml_ids_detect.py#L314) và [`write_html`](ml_ids_detect.py#L330))

- `summary.csv`: mỗi hàng là một file, cột là toàn bộ metric của 5 nhóm.
- `per_class_f1_TSTR.csv`, `per_class_f1_TRS.csv`: F1 của từng loại attack, để thấy synthetic giúp hoặc làm hại loại nào.
- `report.html`: mỗi nhóm một bảng, có ký hiệu ↑/↓ và giải thích từng cột, để gửi cho người khác.