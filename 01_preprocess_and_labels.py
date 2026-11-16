import pandas as pd
from utils import load_fixed_tables, add_train_rul, cap_rul, basic_summary
from pathlib import Path

NUMERIC_PREFIXES = ("setting", "s")  # settings e sensores

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Coercion: Aplica errors='coerce' para detectar não-numéricos
    if "unit" in df.columns:
        df["unit"] = pd.to_numeric(df["unit"], errors="coerce")
    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")

    # Coerce todos os settings e sensores
    for col in df.columns:
        if col.startswith(NUMERIC_PREFIXES):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Remoção de NaNs CRÍTICOS (unit/cycle)
    critical_before = len(df)
    # Remove linhas onde unit ou cycle falharam a conversão
    df = df.dropna(subset=["unit", "cycle"]).copy()
    critical_after = len(df)
    if critical_after < critical_before:
        print(f"[AVISO] Removidas {critical_before - critical_after} linhas com unit/cycle inválidos.")

    # 3. Remoção de NaNs em FEATURES (settings/sensores)
    feature_cols = [c for c in df.columns if c.startswith(NUMERIC_PREFIXES)]
    nan_rows_before = len(df)
    # Remove linhas onde qualquer leitura de feature falhou a conversão
    df = df.dropna(subset=feature_cols).copy()
    nan_rows_after = len(df)
    if nan_rows_after < nan_rows_before:
        print(f"[AVISO] Removidas {nan_rows_before - nan_rows_after} linhas com leituras de sensores/settings inválidas (NaN).")
        
    # 4. Final Type Casting
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df

# A função main permanece a mesma, mas se beneficia da nova coerção.
def main():
    SRC_DIR = Path(__file__).resolve().parent
    ROOT = SRC_DIR.parent
    out_dir = ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_fixed, df_test_fixed, _ = load_fixed_tables()

    # 1) Forçar tipos numéricos e remover NaNs
    df_train_fixed = coerce_numeric(df_train_fixed)
    df_test_fixed  = coerce_numeric(df_test_fixed)

    # 2) Criar RUL no treino
    df_train_prep = add_train_rul(df_train_fixed)

    # 3) Capping do RUL
    df_train_prep = cap_rul(df_train_prep, cap=130)

    # 4) Salvar versões preparadas
    train_prep_path = out_dir / "train_prepared.csv"
    test_prep_path  = out_dir / "test_prepared.csv"
    df_train_prep.to_csv(train_prep_path, index=False)
    df_test_fixed.to_csv(test_prep_path, index=False)

    # 5) Resumo descritivo
    summary = basic_summary(df_train_prep)
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    print("Pré-processamento concluído.")
    print(f"→ {train_prep_path} (shape={df_train_prep.shape})")
    print(f"→ {test_prep_path}  (shape={df_test_fixed.shape})")
    print(f"→ {out_dir / 'dataset_summary.csv'}")

if __name__ == "__main__":
    main()

