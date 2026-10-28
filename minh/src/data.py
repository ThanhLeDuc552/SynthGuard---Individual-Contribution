"""Đọc và chuẩn bị dữ liệu. Mỗi dataset được gọi riêng bằng tên: 'unsw_nb15' hoặc 'cicids2017'."""
import numpy as np
import pandas as pd

from .config import DATASETS, RARE_THRESHOLD, SEED


def _clean_columns(df):
    df.columns = df.columns.str.strip()          # CICIDS2017 có khoảng trắng đầu tên cột
    return df


def load_unsw():
    d = DATASETS["unsw_nb15"]["dir"]
    train_f = d / "UNSW_NB15_training-set_cleaned.csv"
    test_f = d / "UNSW_NB15_testing-set_cleaned.csv"
    for f in (train_f, test_f):
        if not f.exists():
            raise FileNotFoundError(f"Không thấy {f}")
    return _clean_columns(pd.read_csv(train_f)), _clean_columns(pd.read_csv(test_f))


def load_cicids():
    d = DATASETS["cicids2017"]["dir"]
    files = sorted(d.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Không thấy file .csv nào trong {d}")
    parts = []
    for f in files:
        df = _clean_columns(pd.read_csv(f, low_memory=False, encoding="latin-1"))
        df["source_file"] = f.name
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df.replace([np.inf, -np.inf], np.nan)  # Flow Bytes/s, Flow Packets/s có inf


def load(name):
    """Trả về (train, test) cho UNSW-NB15, hoặc (df, None) cho CICIDS2017."""
    if name == "unsw_nb15":
        return load_unsw()
    if name == "cicids2017":
        return load_cicids(), None
    raise ValueError(f"Dataset không hợp lệ: {name}")


def find_label_col(df, name):
    lowered = {c.strip().lower(): c for c in df.columns}
    for cand in DATASETS[name]["label_candidates"]:
        if cand in lowered:
            return lowered[cand]
    raise KeyError(f"Không tìm thấy cột nhãn trong {name}. Các cột: {list(df.columns)[:15]}...")


def class_counts(df, label_col, rare_threshold=RARE_THRESHOLD):
    vc = df[label_col].value_counts(dropna=False).rename_axis(label_col).reset_index(name="rows")
    vc["percent"] = (100 * vc["rows"] / vc["rows"].sum()).round(3)
    vc["rare"] = vc["rows"] < rare_threshold
    return vc


def prepare_test(test_df, name, columns):
    """Chuẩn bị tập kiểm thử THẬT cho IDS: cùng cột với bảng huấn luyện, không giới hạn số dòng.

    `columns` là danh sách cột của bảng huấn luyện (đã qua prepare_for_ctgan).
    """
    df = test_df.replace([np.inf, -np.inf], np.nan)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Tập kiểm thử thiếu cột: {missing}")
    return df[list(columns)].dropna().reset_index(drop=True)


def prepare_for_ctgan(df, name, cap_per_class=3000, seed=SEED):
    """Chuẩn bị bảng cho CTGAN.

    - bỏ cột không dùng, bỏ dòng thiếu/vô cực
    - giữ NGUYÊN các lớp hiếm, chỉ giảm các lớp lớn xuống `cap_per_class` dòng (để huấn luyện nhanh)
    - trả về (bảng, tên cột nhãn, danh sách cột rời rạc)
    """
    label_col = find_label_col(df, name)
    drop = [c for c in DATASETS[name]["drop"] if c in df.columns and c != label_col]
    df = df.drop(columns=drop)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if cap_per_class:
        parts = [g.sample(min(len(g), cap_per_class), random_state=seed)
                 for _, g in df.groupby(label_col)]
        df = pd.concat(parts).reset_index(drop=True)

    discrete = [label_col]
    for c in df.columns:
        if c == label_col:
            continue
        if df[c].dtype == object or df[c].dtype.name in ("string", "category", "str") or df[c].nunique() <= 10:
            discrete.append(c)
    return df, label_col, discrete
