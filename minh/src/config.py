"""Cấu hình chung cho pipeline AI (CTGAN)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"

# Mỗi dataset: nơi đặt file, danh sách tên cột nhãn có thể có (ưu tiên từ trái sang phải),
# và các cột không đưa vào mô hình sinh.
DATASETS = {
    "unsw_nb15": {
        "dir": DATA_DIR / "unsw_nb15",
        "label_candidates": ["attack_cat", "label"],
        "drop": ["id", "label", "source_file"],   # 'label' (0/1) thừa vì đã có attack_cat
    },
    "cicids2017": {
        "dir": DATA_DIR / "cicids2017",
        "label_candidates": ["label"],
        "drop": ["source_file"],
    },
}

RARE_THRESHOLD = 1000   # lớp có ít hơn số bản ghi này bị coi là 'hiếm'
SEED = 42
