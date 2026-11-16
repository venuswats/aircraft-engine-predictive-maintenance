from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from utils import feature_columns
import random

# ---------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------
def load_scaler_stats(path_json: Path):
    """Carrega a média e escala do StandardScaler salvas em JSON."""
    with open(path_json, "r") as f:
        d = json.load(f)
    mean = np.array(d["mean"], dtype=float)
    scale = np.array(d["scale"], dtype=float)
    return mean, scale

def transform(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Aplica a normalização padrão (X - mean) / scale."""
    return (X - mean) / scale

def add_true_rul_to_full_test(df_test: pd.DataFrame, df_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a RUL real para cada ciclo do conjunto de teste,
    usando os ciclos máximos do teste e os valores de Ground Truth.
    """
    # 1. Obtendo o ciclo máximo observado no TESTE para cada unidade
    max_cycle_observed = df_test.groupby('unit')['cycle'].max().reset_index()
    max_cycle_observed.columns = ['unit', 'max_cycle_observed']

    # 2. Adicionando o Ground Truth (RUL_final)
    # A RUL final no PM_truth é a RUL restante A PARTIR do último ciclo observado.
    df_truth_temp = df_truth.copy()
    df_truth_temp.columns = ['unit', 'rul_at_max_cycle']
    
    # Adicionar o valor de RUL do ground truth ao df_test
    df_max_rul = pd.merge(max_cycle_observed, df_truth_temp, on='unit', how='left')

    # 3. Calcular o FIM DA VIDA ÚTIL (EOL - End of Life) para cada unidade
    # EOL = Ciclo_Máximo_Observado + RUL_Final
    df_max_rul['EOL'] = df_max_rul['max_cycle_observed'] + df_max_rul['rul_at_max_cycle']
    
    # 4. Calcular RUL real para CADA ciclo
    df_merged = pd.merge(df_test, df_max_rul[['unit', 'EOL']], on='unit', how='left')
    df_merged['rul_true'] = df_merged['EOL'] - df_merged['cycle']
    
    # 5. Aplicar CAPPING (Se o modelo foi treinado com capping de 130)
    # Embora o capping seja aplicado no treino, alguns modelos aplicam o cap no resultado do RUL real/previsto.
    # Vamos manter o RUL real, mas se precisar de capping (para visualização), adicione:
    # df_merged['rul_true'] = df_merged['rul_true'].clip(upper=130)
    
    return df_merged.dropna(subset=['rul_true'])

def plot_rul_comparison(df_results: pd.DataFrame, figs_dir: Path):
    """Gera o gráfico de comparação das curvas RUL reais e preditas."""
    
    # Seleciona uma amostra aleatória de 6 motores para visualização
    all_units = df_results["unit"].unique()
    sample_units = random.sample(list(all_units), min(6, len(all_units)))
    
    num_plots = len(sample_units)
    cols = 2
    rows = (num_plots + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    plt.suptitle("Comparação de RUL Real vs. RUL Predita (Motores Amostrados)", fontsize=16, y=1.02)
    
    for i, unit_id in enumerate(sample_units):
        df_motor = df_results[df_results["unit"] == unit_id]
        
        plt.subplot(rows, cols, i + 1)
        
        # Plota a curva de RUL REAL
        plt.plot(df_motor["cycle"], df_motor["rul_true"], 
                 label="RUL Real", color="#FF5733", linewidth=2)
        
        # Plota a curva de RUL PREVISTA (apenas para o último ciclo, para simular a previsão real)
        # OU plotamos toda a curva de previsão, o que é mais informativo.
        plt.plot(df_motor["cycle"], df_motor["rul_pred"], 
                 label="RUL Predita (MLP)", color="#00A79D", linestyle='--', linewidth=2)
        
        plt.title(f"Motor ID: {int(unit_id)}")
        plt.xlabel("Ciclo de Operação")
        plt.ylabel("RUL (Ciclos)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout(rect=[0, 0, 1, 1])
    plot_path = figs_dir / "rul_comparison_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Figura de Comparação de RUL salva em: {plot_path}")


# ------------------------------------------------
# Execução principal (Modificada)
# ---------------------------------------------------
def main():
    ROOT = Path(__file__).resolve().parent.parent
    tables_dir = ROOT / "outputs" / "tables"
    figs_dir = ROOT / "outputs" / "figures"
    models_dir = ROOT / "outputs" / "models"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # Carregar dados e modelo
    # ---------------------------------------------------
    df_train = pd.read_csv(tables_dir / "train_prepared.csv")
    df_test  = pd.read_csv(tables_dir / "test_prepared.csv")
    # Carrega Ground Truth e padroniza
    df_truth = pd.read_csv(ROOT / "PM_truth.csv", header=None, names=["unit", "RUL"]) 
    df_truth['unit'] = df_truth.index.map(lambda i: i + 1) # Adiciona coluna de ID (1 a 100)
    df_truth = df_truth.rename(columns={'unit': 'unit_id', 'RUL': 'rul_at_max_cycle'})

    feats = feature_columns(df_train)

    # Modelo e scaler
    model = tf.keras.models.load_model(models_dir / "mlp.h5", compile=False)
    mean, scale = load_scaler_stats(models_dir / "scaler_stats.json")

    # ---------------------------------------------------
    # 1. Preparar dados para o CONJUNTO COMPLETO de TESTE (para Plotagem)
    # ---------------------------------------------------
    df_full_test = df_test.copy()
    
    # 1.1 Calcular a RUL verdadeira para TODAS as linhas de df_full_test
    df_full_test = add_true_rul_to_full_test(df_full_test, df_truth)

    # 1.2 Fazer a predição para TODAS as linhas
    X_full_te = df_full_test[feats].values
    X_full_te_s = transform(X_full_te, mean, scale)
    y_full_pred = model.predict(X_full_te_s).ravel()
    df_full_test["rul_pred"] = y_full_pred

    # ---------------------------------------------------
    # 2. Calcular métricas (Apenas no ÚLTIMO CICLO de cada motor)
    # ---------------------------------------------------
    # Isolamos o último ciclo para o cálculo das métricas
    last_idx = df_full_test.groupby("unit")["cycle"].idxmax()
    df_last_cycle = df_full_test.loc[last_idx].reset_index(drop=True)

    y_true_final = df_last_cycle["rul_true"].values
    y_pred_final = df_last_cycle["rul_pred"].values

    # Calcular e salvar métricas
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    mae  = mean_absolute_error(y_true_final, y_pred_final)
    r2   = r2_score(y_true_final, y_pred_final)

    test_metrics = pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2}])
    test_metrics.to_csv(tables_dir / "test_metrics.csv", index=False)

    # ---------------------------------------------------
    # 3. Gerar figuras (Plotagem)
    # ---------------------------------------------------
    # Chama a nova função para o gráfico de comparação de RUL
    plot_rul_comparison(df_full_test, figs_dir)
    
    # Parity plot (RUL real vs predito) - Usando os valores do último ciclo
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true_final, y_pred_final, s=12)
    mn, mx = float(y_true_final.min()), float(y_true_final.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
    plt.xlabel("RUL real (Ciclos)")
    plt.ylabel("RUL predita (Ciclos)")
    plt.title("Parity Plot - Último Ciclo de Teste")
    plt.tight_layout()
    plt.savefig(figs_dir / "parity_plot.png", dpi=200)
    plt.close()

    # Histograma de resíduos - Usando os valores do último ciclo
    resid = y_pred_final - y_true_final
    plt.figure(figsize=(6, 5))
    plt.hist(resid, bins=20, color="#4682B4", edgecolor="black")
    plt.xlabel("Resíduo (pred - real)")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos Resíduos - Último Ciclo")
    plt.tight_layout()
    plt.savefig(figs_dir / "residuals_hist.png", dpi=200)
    plt.close()
    
    # Tabela de erros por unidade (mantida)
    per_unit = pd.DataFrame({
        "unit": df_last_cycle["unit"].values,
        "rul_true": y_true_final,
        "rul_pred": y_pred_final,
        "abs_err": np.abs(y_pred_final - y_true_final)
    }).sort_values("abs_err", ascending=False)
    per_unit.to_csv(tables_dir / "per_unit_errors.csv", index=False)

    # ---------------------------------------------------
    # Logs finais
    # ---------------------------------------------------
    print("Avaliação concluída.")
    print(f"TESTE → RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
    print(f"Figuras salvas em: {figs_dir}")
    print(f"Tabelas salvas em: {tables_dir}")

if __name__ == "__main__":
    main()