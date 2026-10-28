"""AI1 (CTGAN) + AI3 (sinh dữ liệu theo quota từng lớp) + kiểm tra nhanh chất lượng."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def train_ctgan(df, discrete_columns, epochs=300, batch_size=500, pac=10, cuda=False, seed=42):
    """Huấn luyện CTGAN. batch_size phải chia hết cho pac."""
    import torch
    from ctgan import CTGAN
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CTGAN(epochs=epochs, batch_size=batch_size, pac=pac, cuda=cuda, verbose=True)
    model.fit(df, discrete_columns)
    return model


def train_tvae(df, discrete_columns, epochs=300, batch_size=500, cuda=False, seed=42):
    """AI-2: huấn luyện TVAE (cùng thư viện `ctgan`, cùng cách mã hóa dữ liệu bảng như CTGAN)."""
    import torch
    from ctgan import TVAE
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TVAE(epochs=epochs, batch_size=batch_size, cuda=cuda, verbose=True)
    model.fit(df, discrete_columns)
    return model


def sample_pool_by_class(model, label_col, quotas, batch=50_000, max_rounds=20, verbose=True):
    """Sinh theo quota từng lớp cho mô hình KHÔNG có điều kiện lớp (TVAE).

    TVAE không nhận điều kiện nhãn, nên sinh từng lô lớn rồi giữ các dòng thuộc lớp cần.
    Lớp hiếm xuất hiện ít trong lô nên có thể không đủ quota; hàm in cảnh báo thay vì độn thêm.
    Đây cũng là một khác biệt thực tế giữa TVAE và CTGAN (CTGAN có điều kiện lớp).
    """
    need = dict(quotas)
    kept = {c: [] for c in quotas}
    drawn = 0
    hits = {c: 0 for c in quotas}
    for _ in range(max_rounds):
        if all(v <= 0 for v in need.values()):
            break
        pool = model.sample(batch)
        drawn += len(pool)
        for cls in quotas:
            if need[cls] <= 0:
                continue
            m = pool[pool[label_col] == cls]
            hits[cls] += len(m)
            take = m.head(need[cls])
            kept[cls].append(take)
            need[cls] -= len(take)
    parts = [pd.concat(v, ignore_index=True) for v in kept.values() if v]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if verbose:
        rep = pd.DataFrame([(c, quotas[c], quotas[c] - max(need[c], 0), hits[c], hits[c] / drawn if drawn else 0.0)
                            for c in quotas],
                           columns=["class", "quota", "got", "seen_in_pool", "share_of_pool"])
        print(f"Đã sinh {drawn:,} dòng trong pool")
        print(rep.to_string(index=False))
        short = rep[rep["got"] < rep["quota"]]
        if len(short):
            print("CẢNH BÁO: các lớp thiếu so với quota:", list(short["class"]))
    return out


def sample_by_class(model, label_col, quotas, max_rounds=10, oversample=2.0, verbose=True):
    """Sinh dữ liệu có điều kiện theo nhãn. quotas = {'Worms': 500, 'Shellcode': 500, ...}

    CTGAN chỉ dùng điều kiện lớp như một gợi ý: nhãn trong dòng sinh ra có thể khác lớp được
    yêu cầu (nhất là khi chưa huấn luyện đủ). Vì vậy sinh dư, chỉ giữ dòng có nhãn đúng lớp
    (rejection sampling), lặp tối đa max_rounds vòng. Nếu vẫn thiếu thì trả về ít hơn quota và
    in cảnh báo, không âm thầm gán nhãn lại.
    """
    parts, report = [], []
    for cls, n in quotas.items():
        kept, drawn, rounds, hits = [], 0, 0, 0
        need = n
        while need > 0 and rounds < max_rounds:
            # tỷ lệ khớp quan sát được (vòng đầu: giả định 50%); sinh đủ để bù, tối đa 50k dòng/vòng
            rate = hits / drawn if drawn else 0.5
            size = int(min(50_000, np.ceil(need * max(oversample, 1.5 / max(rate, 0.005))) + 10))
            batch = model.sample(size, condition_column=label_col, condition_value=cls)
            drawn += len(batch)
            match = batch[batch[label_col] == cls]
            hits += len(match)
            kept.append(match.head(need))
            need = n - sum(len(k) for k in kept)
            rounds += 1
        got = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
        report.append((cls, n, len(got), drawn, len(got) / drawn if drawn else 0.0))
        parts.append(got)
    if verbose:
        rep = pd.DataFrame(report, columns=["class", "quota", "got", "drawn", "match_rate"])
        print(rep.to_string(index=False))
        short = rep[rep["got"] < rep["quota"]]
        if len(short):
            print("CẢNH BÁO: các lớp thiếu so với quota:", list(short["class"]))
    return pd.concat(parts, ignore_index=True)


def quick_fidelity(real, synth, discrete_columns):
    """Kiểm tra nhanh: KS (cột số) và total variation distance (cột rời rạc). Càng gần 0 càng giống."""
    rows = []
    for c in real.columns:
        if c in discrete_columns:
            p = real[c].value_counts(normalize=True)
            q = synth[c].value_counts(normalize=True)
            idx = p.index.union(q.index)
            tv = 0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum()
            rows.append((c, "discrete", float(tv)))
        else:
            rows.append((c, "numeric", float(ks_2samp(real[c], synth[c]).statistic)))
    out = pd.DataFrame(rows, columns=["column", "type", "distance"])
    return out.sort_values("distance", ascending=False).reset_index(drop=True)
