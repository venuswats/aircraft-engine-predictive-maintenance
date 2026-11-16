from pathlib import Path
import numpy as np
import pandas as pd
import random
import os

def set_reproducible(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def load_fixed_tables():
    """
    Lê os CSVs já corrigidos (com cabeçalhos) a partir de outputs/tables
    e o PM_truth.csv na raiz do projeto.
    """
    SRC_DIR = Path(__file__).resolve().parent
    ROOT = SRC_DIR.parent

    train_fixed = ROOT / "outputs" / "tables" / "PM_train_fixed.csv"
    test_fixed  = ROOT / "outputs" / "tables" / "PM_test_fixed.csv"
    truth_path  = ROOT / "PM_truth.csv"

    df_train = pd.read_csv(train_fixed)
    df_test  = pd.read_csv(test_fixed)
    df_truth = pd.read_csv(truth_path, header=None, names=["RUL"])

    
    expected_first_cols = ["unit", "cycle", "setting1", "setting2", "setting3"]
    assert all(c in df_train.columns[:5] for c in expected_first_cols), "Cabeçalhos inesperados no train_fixed."
    assert all(c in df_test.columns[:5]  for c in expected_first_cols),  "Cabeçalhos inesperados no test_fixed."

    return df_train, df_test, df_truth

def add_train_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    max_cycle_per_unit = df_train.groupby("unit")["cycle"].transform("max")
    df = df_train.copy()
    df["RUL"] = (max_cycle_per_unit - df["cycle"]).astype(int)
    return df

def cap_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    if cap is None:
        return df
    out = df.copy()
    out["RUL"] = out["RUL"].clip(upper=cap)
    return out

def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    info = {
        "num_units": [df["unit"].nunique()],
        "num_rows": [len(df)],
        "num_cols": [df.shape[1]],
        "min_cycles_per_unit": [df.groupby("unit")["cycle"].max().min()],
        "max_cycles_per_unit": [df.groupby("unit")["cycle"].max().max()],
    }
    return pd.DataFrame(info)

def feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna apenas as colunas de features (remove unit, cycle e RUL, se existir)."""
    drop = {"unit", "cycle", "RUL"}
    return [c for c in df.columns if c not in drop]
