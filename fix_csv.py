from pathlib import Path
import pandas as pd

def make_column_names(n_cols: int):
    """
    Gera nomes: unit, cycle, setting1..setting3, s1..sN
    (assumindo formato típico do C-MAPSS derivado).
    """
    if n_cols < 5:
        raise ValueError(
            f"Número inesperado de colunas ({n_cols}). "
            "Verifique o separador do CSV (vírgula)."
        )
    names = ["unit", "cycle", "setting1", "setting2", "setting3"]
    sensor_cols = [f"s{i}" for i in range(1, n_cols - 5 + 1)]
    return names + sensor_cols

def main():
    # Caminhos base
    SRC_DIR = Path(__file__).resolve().parent
    ROOT = SRC_DIR.parent  # tcc/
    
    # Arquivos de entrada 
    train_path = ROOT / "PM_train.csv"
    test_path  = ROOT / "PM_test.csv"

    # Saída
    out_dir = ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "PM_train_fixed.csv"
    test_out  = out_dir / "PM_test_fixed.csv"

    # Leitura com vírgula como separador
    df_train = pd.read_csv(train_path, sep=",", header=None)
    df_test  = pd.read_csv(test_path,  sep=",", header=None)

    # Nomeação de colunas
    df_train.columns = make_column_names(df_train.shape[1])
    df_test.columns  = make_column_names(df_test.shape[1])

    # Salvando CSVs corrigidos
    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)

    # Feedback ao usuário
    print(f"→ {train_out}  (shape={df_train.shape})")
    print(f"→ {test_out}   (shape={df_test.shape})")

if __name__ == "__main__":
    main()

