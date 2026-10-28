"""AI-4: IDS baseline.

Cùng một mô hình (Random Forest, cùng tham số, cùng seed) được huấn luyện trên các bộ dữ liệu khác nhau,
rồi đánh giá trên cùng MỘT tập kiểm thử thật chưa được dùng để huấn luyện (kể cả generator).
Chỉ thay đổi dữ liệu huấn luyện, nên chênh lệch kết quả đến từ dữ liệu sinh.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import OneHotEncoder

from .config import SEED
from .evaluation import as_cat


def _frame(df, features, cat):
    x = pd.DataFrame(index=df.index)
    for c in features:
        x[c] = as_cat(df[c]) if c in cat else pd.to_numeric(df[c], errors="coerce").astype(float)
    return x


def evaluate_ids(train_df, test_df, label_col, discrete, seed=SEED, n_estimators=200):
    """Huấn luyện Random Forest trên train_df, đánh giá trên test_df.

    Trả về (dict số liệu tổng, bảng kết quả theo từng lớp).
    """
    features = [c for c in train_df.columns if c != label_col]
    cat = [c for c in discrete if c != label_col]

    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                             remainder="passthrough")
    Xtr = prep.fit_transform(_frame(train_df, features, cat))
    Xte = prep.transform(_frame(test_df, features, cat))
    ytr = train_df[label_col].astype(str).to_numpy()
    yte = test_df[label_col].astype(str).to_numpy()

    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    proba = rf.predict_proba(Xte)

    classes = sorted(set(yte) | set(ytr))
    p, r, f, sup = precision_recall_fscore_support(yte, pred, labels=classes, zero_division=0)
    pr_auc = []
    for c in classes:
        pos = (yte == c)
        if c in rf.classes_ and pos.any():
            pr_auc.append(average_precision_score(pos, proba[:, list(rf.classes_).index(c)]))
        else:
            pr_auc.append(np.nan)

    per_class = pd.DataFrame({"class": classes, "test_support": sup, "precision": p, "recall": r,
                              "f1": f, "pr_auc": pr_auc})
    in_test = per_class["test_support"] > 0
    overall = {
        "accuracy": accuracy_score(yte, pred),
        "macro_precision": per_class.loc[in_test, "precision"].mean(),
        "macro_recall": per_class.loc[in_test, "recall"].mean(),
        "macro_f1": per_class.loc[in_test, "f1"].mean(),
        "macro_pr_auc": per_class.loc[in_test, "pr_auc"].mean(),
    }
    return overall, per_class


def run_experiments(experiments, test_df, label_col, discrete, seed=SEED, n_estimators=200):
    """experiments = {'Real only': df, 'Real + CTGAN': df, ...}.

    Trả về (bảng tổng hợp, bảng recall theo lớp: mỗi cột một thí nghiệm).
    """
    rows, recalls = [], {}
    for name, train_df in experiments.items():
        overall, per_class = evaluate_ids(train_df, test_df, label_col, discrete, seed, n_estimators)
        rows.append({"experiment": name, "train_rows": len(train_df), **overall})
        recalls[name] = per_class.set_index("class")["recall"]
    summary = pd.DataFrame(rows).set_index("experiment")
    recall_by_class = pd.DataFrame(recalls)
    support = per_class.set_index("class")["test_support"]
    recall_by_class.insert(0, "test_support", support)
    return summary, recall_by_class
