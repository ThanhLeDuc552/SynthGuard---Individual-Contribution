"""ML-IDS test cho dataset NetFlow UNSW-NB15 (49 cot, nhan = cot Attack).

Chay:
  python ids_netflow.py --train REAL_TRAIN.csv --test REAL_TEST.csv \
      --synthetic synthetic_ctgan_unsw_nb15_original.csv synthetic_ctgan_unsw_nb15_smote_10k.csv ...

- Luon chay "real-only" truoc, roi moi file synthetic: train = real + synthetic.
- Test luon la REAL (khong bao gio tron synthetic vao test).
- Synthetic chi co dong tan cong (khong co Normal) nen chi them vao train.
- Cac cot rac/khong dung duoc se bi bo qua; gia tri vo han/NaN duoc thay bang 0.
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

DROP = ["Label", "Attack"]


def load(p):
    d = pd.read_csv(p)
    return d


def xy(d):
    X = d.drop(columns=[c for c in DROP if c in d.columns]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, d["Attack"].astype(str)


def run(tr, te, tag):
    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    Xte = Xte[Xtr.columns]
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                 class_weight="balanced_subsample")
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    print(f"\n===== {tag} (train={len(tr)}) =====")
    print(classification_report(yte, p, digits=3, zero_division=0))
    mf1 = f1_score(yte, p, average="macro", zero_division=0)
    print("macro-F1:", round(mf1, 4))
    return mf1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", help="bo trong thi tu chia --train thanh 80/20 theo lop (seed 42)")
    ap.add_argument("--synthetic", nargs="*", default=[])
    ap.add_argument("--dedup", action="store_true", help="bo dong trung trong train real")
    a = ap.parse_args()
    tr = load(a.train)
    if a.test:
        te = load(a.test)
    else:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(tr, test_size=0.2, random_state=42, stratify=tr["Attack"])
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        print(f"[tu chia] train={len(tr)} test={len(te)}")
    if a.dedup:
        tr = tr.drop_duplicates().reset_index(drop=True)
    res = {"real-only": run(tr, te, "real-only")}
    for s in a.synthetic:
        name = os.path.basename(s).replace("synthetic_ctgan_unsw_nb15_", "").replace(".csv", "")
        syn = load(s)
        syn = syn[[c for c in tr.columns if c in syn.columns]]
        res["real+" + name] = run(pd.concat([tr, syn], ignore_index=True), te, "real + " + name)
    print("\n== TOM TAT macro-F1 ==")
    for k, v in res.items():
        print(f"{k:28s} {v:.4f}")
