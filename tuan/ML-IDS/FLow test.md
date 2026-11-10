Tầng 1 (giống nhau cho mọi cấu hình):
  binary train/fold_k ──► detector benign/attack (supervised + anomaly, như ml_ids.py)

Tầng 2 (4 cấu hình, train trên data đã scale bằng scaler của nhóm):
  A. Original (real-only)    ─┐
  B. Multiclass (real+SMOTE)  ├─► classifier loại attack
  C. Multiclass + CTGAN       │
  D. Multiclass + TVAE       ─┘

Test end-to-end trên binary test/fold_k:
  flow ─► tầng 1: benign? ─► dừng
                 attack? ─► tầng 2: loại attack nào?

Cách so sánh (mỗi phép so trả lời một câu hỏi):
  B vs A  : SMOTE có giúp không?
  C vs B  : CTGAN có giúp thêm so với chỉ dùng multiclass không?
  D vs B  : TVAE có giúp thêm so với chỉ dùng multiclass không?
  C, D vs A : cả gói (SMOTE + generator) có tốt hơn chỉ dùng data thật không?

Nên nhìn: F1 tầng 2 trên các lớp hiếm (Worms, Shellcode, Analysis, Backdoor)
          và tỉ lệ "phát hiện + gán đúng loại" theo từng loại attack (end-to-end).
