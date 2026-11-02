import numpy as np, pandas as pd
from ctgan import CTGAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

LABEL = "Attack"
# columns that are codes, not measurements (only those present in the data are used)
CAT_CANDIDATES = ["PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
                  "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
                  "MIN_TTL", "MAX_TTL"]
ID_COLS = ["DNS_QUERY_ID"]          # random IDs: no information
TOP_K = 30                           # keep the K most common codes, group the rest as "other"
SKEW_LIMIT = 5                       # numeric columns more skewed than this get log1p

def fit_prep(real):
    """Learn everything from the REAL training data only."""
    p = {}
    df = real.replace([np.inf, -np.inf], np.nan)
    p["cat"] = [c for c in CAT_CANDIDATES if c in df.columns]
    p["drop"] = {c: df[c].mode(dropna=True).iloc[0] for c in df.columns
                 if c != LABEL and (c in ID_COLS or df[c].nunique(dropna=True) <= 1
                                    or df[c].value_counts(normalize=True, dropna=True).iloc[0] > 0.999)}
    p["cat"] = [c for c in p["cat"] if c not in p["drop"]]
    p["num"] = [c for c in df.columns if c not in p["cat"] + list(p["drop"]) + [LABEL]]
    p["top"] = {c: set(df[c].value_counts().index[:TOP_K]) for c in p["cat"]}
    p["rare_vals"] = {c: df.loc[~df[c].isin(p["top"][c]), c].dropna().to_numpy() for c in p["cat"]}
    p["log"] = [c for c in p["num"] if df[c].min() >= 0 and df[c].skew() > SKEW_LIMIT]
    p["median"] = df[p["num"]].median()
    p["max"] = df[p["num"]].max()
    p["is_int"] = {c: bool(np.all(np.isclose(df[c].dropna() % 1, 0))) for c in p["num"]}
    p["order"] = list(real.columns)
    return p

def _as_str(series):
    """Code column -> clean text ('6', not '6.0'); works for integer, non-integer and missing values."""
    if series.dtype.kind in "iuf":
        return series.map(lambda v: "nan" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v))).to_numpy()
    return series.astype(str).to_numpy()

def transform(df, p):
    df = df.replace([np.inf, -np.inf], np.nan)
    out = df.drop(columns=list(p["drop"])).copy()
    for c in p["num"]:
        out[c] = out[c].fillna(p["median"][c])
        if c in p["log"]: out[c] = np.log1p(out[c].clip(lower=0))
    for c in p["cat"]:
        out[c] = np.where(out[c].isin(p["top"][c]), _as_str(out[c]), "other")
    return out

def inverse(syn, p, seed=0):
    rng = np.random.default_rng(seed)
    out = syn.copy()
    for c in p["num"]:
        if c in p["log"]: out[c] = np.expm1(out[c])
        out[c] = out[c].clip(lower=0, upper=p["max"][c])           # no negatives, no values above the real max
        if p["is_int"][c]: out[c] = out[c].round()
    for c in p["cat"]:
        m = out[c] == "other"
        if m.any() and len(p["rare_vals"][c]):
            out.loc[m, c] = rng.choice(p["rare_vals"][c], size=int(m.sum()))
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c, v in p["drop"].items(): out[c] = v                      # constant / ID columns put back
    return out[p["order"]]

def label_consistency(real, syn, seed=0):
    """Train on REAL, predict the type of each SYNTHETIC row; share that matches the label the generator gave it."""
    feats = [c for c in real.columns if c != LABEL]
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    clean = lambda d: d[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    rf.fit(clean(real), real[LABEL])
    pred = rf.predict(clean(syn))
    ok = pd.Series(pred == syn[LABEL].to_numpy(), index=syn.index)
    return ok, ok.groupby(syn[LABEL]).mean().round(3)

def cat_validity(real, syn, p):
    return pd.Series({c: round(float(syn[c].isin(set(real[c].dropna())).mean()), 3) for c in p["cat"]})

def sample_matched(model, p, real, seed=0, max_rounds=8):
    """Sample from the ONE trained CTGAN with an explicit condition on the label, keeping only rows whose
    generated label really equals the requested one, until each class has the share it has in the
    training file. Labels are never overwritten: if the generator ignores the condition the class
    comes out short and the shortfall is printed."""
    parts, short = [], {}
    for cls, n in real[LABEL].value_counts().items():
        n, have = int(n), 0
        for _ in range(max_rounds):
            s = model.sample(max((n - have) * 2, 100), condition_column=LABEL, condition_value=cls)
            s = s[s[LABEL] == cls].head(n - have)
            parts.append(s); have += len(s)
            if have >= n: break
        if have < n: short[cls] = n - have
    if short: print("  rows missing (generator ignored the condition):", short)
    syn = inverse(pd.concat(parts, ignore_index=True), p, seed)
    return syn.sample(frac=1, random_state=seed).reset_index(drop=True)

def train_and_sample(real, epochs=300, batch_size=500, pac=10, embedding_dim=128, seed=42,
                     verbose=True, matched=True, **ctgan_kwargs):
    """ONE CTGAN for the whole dataset. ctgan_kwargs: generator_dim, discriminator_dim, generator_lr,
    discriminator_lr, discriminator_steps, log_frequency, ... (anything CTGAN accepts)."""
    import torch; torch.manual_seed(seed)
    p = fit_prep(real)
    X = transform(real, p)
    step = pac if pac % 2 == 0 else pac * 2                       # CTGAN needs an even batch size that is a multiple of pac
    bs = max(step * 2, min(batch_size, (len(X) // step) * step))   # ...and not larger than the data
    model = CTGAN(epochs=epochs, batch_size=bs, pac=pac, embedding_dim=embedding_dim, verbose=verbose, **ctgan_kwargs)
    model.fit(X, discrete_columns=p["cat"] + [LABEL])
    syn = sample_matched(model, p, real, seed) if matched else inverse(model.sample(len(real)), p, seed)
    return model, p, syn

def evaluate(real_train, real_test, syn, seed=0):
    """Compound score for tuning. The generator must NOT have seen real_test.
    1) label consistency (macro): RF trained on real_train predicts the type of each synthetic row
    2) TSTR macro-F1: RF trained on synthetic data, tested on the unseen real_test
    score = mean of the two (rare classes count as much as big ones)."""
    feats = [c for c in real_train.columns if c != LABEL]
    clean = lambda d: d[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    ok, per_class = label_consistency(real_train, syn, seed)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(clean(syn), syn[LABEL])
    tstr = f1_score(real_test[LABEL], rf.predict(clean(real_test)), average="macro", zero_division=0)
    lc = float(per_class.mean())
    return {"label_consistency_macro": round(lc, 4), "tstr_macro_f1": round(float(tstr), 4),
            "score": round((lc + float(tstr)) / 2, 4)}, per_class
