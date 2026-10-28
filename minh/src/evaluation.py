"""AI-3: kiểm tra chất lượng dữ liệu sinh.

- validity_report   : giá trị có hợp lệ không (ngoài khoảng của dữ liệu thật, nhãn/loại chưa từng thấy)
- duplicate_report  : bao nhiêu dòng sinh ra trùng hệt dòng huấn luyện / trùng nhau
- dcr_report        : khoảng cách tới bản ghi huấn luyện gần nhất (DCR), so với tập thật chưa dùng để huấn luyện
- class_table       : số bản ghi từng lớp, thật so với sinh
- invalid_row_share / clip_to_real_range : tỷ lệ dòng có giá trị không thể có, và hậu xử lý kéo về khoảng dữ liệu thật

Lưu ý: đây là kiểm tra sơ bộ của nhóm AI. Fidelity chính thức (DS3) và đánh giá MIA (CY2/CY5)
thuộc nhóm khác, và phải dùng tập holdout do DS2 quy định.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def as_cat(s):
    """Chuyển cột rời rạc về chuỗi nhất quán (1, 1.0 và '1' đều thành '1')."""
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        if (v.dropna() % 1 == 0).all():
            return v.astype("Int64").astype(str)
    return s.astype(str)


def class_table(real, synth, label_col):
    t = pd.DataFrame({"thật": real[label_col].value_counts(),
                      "sinh": synth[label_col].value_counts()}).fillna(0).astype(int)
    t["sinh/thật"] = (t["sinh"] / t["thật"].replace(0, np.nan)).round(2)
    return t.sort_values("thật")


def validity_report(real, synth, discrete):
    """Tỷ lệ dòng sinh ra có giá trị không hợp lệ theo từng cột.

    - cột số: nằm ngoài [min, max] của dữ liệu thật
    - cột rời rạc: có giá trị chưa từng xuất hiện trong dữ liệu thật
    """
    rows = []
    for c in real.columns:
        if c in discrete:
            seen = set(as_cat(real[c]))
            bad = ~as_cat(synth[c]).isin(seen)
            rows.append((c, "discrete", float(bad.mean())))
        else:
            lo, hi = real[c].min(), real[c].max()
            bad = (synth[c] < lo) | (synth[c] > hi)
            rows.append((c, "numeric", float(bad.mean())))
    out = pd.DataFrame(rows, columns=["column", "type", "invalid_share"])
    return out.sort_values("invalid_share", ascending=False).reset_index(drop=True)


def _row_hashes(df, columns, discrete, decimals=6):
    x = pd.DataFrame(index=df.index)
    for c in columns:
        if c in discrete:
            x[c] = as_cat(df[c])
        else:
            x[c] = pd.to_numeric(df[c], errors="coerce").astype(float).round(decimals)
    return pd.util.hash_pandas_object(x, index=False)


def duplicate_report(train, synth, discrete, decimals=6):
    """Dòng sinh ra trùng hệt dòng huấn luyện (sao chép) hoặc trùng nhau."""
    cols = list(train.columns)
    h_train = _row_hashes(train, cols, discrete, decimals)
    h_synth = _row_hashes(synth, cols, discrete, decimals)
    return pd.Series({
        "synth_copy_of_train_share": float(h_synth.isin(set(h_train)).mean()),
        "synth_internal_duplicate_share": float(h_synth.duplicated().mean()),
        "train_internal_duplicate_share": float(h_train.duplicated().mean()),
    }, name="share")


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _encoder(columns, discrete):
    num = [c for c in columns if c not in discrete]
    cat = [c for c in columns if c in discrete]
    return ColumnTransformer([
        ("num", Pipeline([("log", FunctionTransformer(_signed_log)), ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def _to_frame(df, columns, discrete):
    x = pd.DataFrame(index=df.index)
    for c in columns:
        x[c] = as_cat(df[c]) if c in discrete else pd.to_numeric(df[c], errors="coerce").astype(float)
    return x


def dcr_report(train, synth, holdout, discrete, max_n=5000, max_train=30000, seed=42):
    """Distance to Closest Record: mỗi dòng sinh ra cách dòng huấn luyện gần nhất bao xa?

    Tham chiếu: khoảng cách từ các dòng THẬT chưa dùng huấn luyện (holdout) tới tập huấn luyện.
    Nếu dòng sinh ra gần tập huấn luyện hơn hẳn holdout thật thì có dấu hiệu mô hình nhớ/chép dữ liệu.
    Trả về (bảng tóm tắt, mảng DCR của dữ liệu sinh, mảng DCR của holdout).
    """
    cols = list(train.columns)
    rng = np.random.default_rng(seed)

    def sub(df, n):
        return df.iloc[rng.choice(len(df), min(n, len(df)), replace=False)]

    tr, sy, ho = sub(train, max_train), sub(synth, max_n), sub(holdout, max_n)
    enc = _encoder(cols, discrete)
    Xtr = enc.fit_transform(_to_frame(tr, cols, discrete))
    Xsy = enc.transform(_to_frame(sy, cols, discrete))
    Xho = enc.transform(_to_frame(ho, cols, discrete))

    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(Xtr)
    d_sy = nn.kneighbors(Xsy)[0][:, 0]
    d_ho = nn.kneighbors(Xho)[0][:, 0]

    p5_ho = np.percentile(d_ho, 5)
    summary = pd.DataFrame({
        "synthetic": [np.median(d_sy), np.percentile(d_sy, 5), d_sy.min(), float((d_sy <= 1e-9).mean()),
                      float((d_sy < p5_ho).mean())],
        "real_holdout": [np.median(d_ho), p5_ho, d_ho.min(), float((d_ho <= 1e-9).mean()), 0.05],
    }, index=["median_dcr", "p5_dcr", "min_dcr", "share_exact_match", "share_closer_than_holdout_p5"])
    return summary, d_sy, d_ho


def invalid_row_share(real, synth, discrete):
    """Tỷ lệ dòng sinh ra có ÍT NHẤT MỘT giá trị không hợp lệ (ngoài khoảng thật / loại chưa từng thấy)."""
    bad = pd.Series(False, index=synth.index)
    for c in real.columns:
        if c in discrete:
            bad |= ~as_cat(synth[c]).isin(set(as_cat(real[c])))
        else:
            bad |= (synth[c] < real[c].min()) | (synth[c] > real[c].max())
    return float(bad.mean())


def clip_to_real_range(real, synth, discrete):
    """Hậu xử lý: kéo cột số về [min, max] của dữ liệu thật và làm tròn các cột số nguyên.

    Chỉ sửa giá trị bất khả thi (ví dụ độ dài kết nối hoặc số gói tin âm). Đây là bước xử lý hậu kỳ,
    nên báo cáo kết quả cả khi có và không có bước này.
    """
    out = synth.copy()
    for c in real.columns:
        if c in discrete:
            continue
        out[c] = out[c].clip(real[c].min(), real[c].max())
        if pd.api.types.is_integer_dtype(real[c]):
            out[c] = out[c].round().astype(real[c].dtype)
    return out
