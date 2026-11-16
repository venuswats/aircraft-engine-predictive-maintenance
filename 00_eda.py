from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import feature_columns

# Função auxiliar para remover colunas constantes (repetida para garantir consistência)
def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas que têm um único valor único (constante)."""
    # Exclui IDs e RUL
    cols_to_check = feature_columns(df)
    
    # Adicionamos 'setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19' 
    # como constantes conhecidas no FD001 (caso a filtragem falhe)
    CONSTANT_COLS = ["setting3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]

    df_out = df.drop(columns=CONSTANT_COLS, errors='ignore')
    
    # Filtra colunas que ficaram constantes após a remoção acima
    initial_features = feature_columns(df_out)
    constant_cols_after_drop = [col for col in initial_features if df_out[col].nunique(dropna=False) <= 1]
    df_out = df_out.drop(columns=constant_cols_after_drop, errors='ignore')

    return df_out

def main():
    # Caminhos
    ROOT = Path(__file__).resolve().parent.parent
    tables_dir = ROOT / "outputs" / "tables"
    figs_dir = ROOT / "outputs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados preparados (assumindo que 01_preprocess_and_labels.py já rodou)
    train_prep_path = tables_dir / "train_prepared.csv"
    if not train_prep_path.exists():
        print(f"Erro: Arquivo {train_prep_path} não encontrado. Execute 01_preprocess_and_labels.py.")
        return

    df_train = pd.read_csv(train_prep_path)
    
    # 1. Filtro de Features Constantes (Necessário para todas as análises)
    df_train_filtered = drop_constant_features(df_train)
    FEATURES = feature_columns(df_train_filtered)
    
    # Sensores críticos para visualização
    CRITICAL_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s11", "s12", "s13", "s14", "s15"]
    
    print(f"Features finais para análise: {FEATURES}")
    
    # ---------------------------------------------------
    # 2. Geração de Métricas Estatísticas Descritivas
    # ---------------------------------------------------
    print("\n--- Gerando Métricas Estatísticas Descritivas ---")
    df_desc = df_train_filtered[FEATURES].describe().transpose()
    df_desc.to_csv(tables_dir / "descriptive_stats.csv")
    print(f"Tabela de estatísticas descritivas salva em: {tables_dir / 'descriptive_stats.csv'}")

    # ---------------------------------------------------
    # 3. Geração de Visualizações (Boxplots e Histogramas)
    # ---------------------------------------------------
    
    # A. Boxplots para Visualizar Escala e Outliers
    plt.figure(figsize=(18, 12))
    df_plot = df_train_filtered[CRITICAL_SENSORS]
    
    for i, col in enumerate(df_plot.columns):
        plt.subplot(3, 4, i + 1)
        sns.boxplot(y=df_plot[col], color='skyblue')
        plt.title(f"Boxplot: {col}", fontsize=10)
        plt.ylabel("") # Remove o rótulo Y repetido
        
    plt.suptitle("Análise de Variabilidade e Outliers (Boxplots)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(figs_dir / "eda_boxplots.png", dpi=200)
    plt.close()
    print(f"Boxplots salvos em: {figs_dir / 'eda_boxplots.png'}")

    # B. Histogramas para Visualizar Distribuição
    plt.figure(figsize=(18, 12))
    
    for i, col in enumerate(df_plot.columns):
        plt.subplot(3, 4, i + 1)
        sns.histplot(df_plot[col], kde=True, bins=30, color='#00A79D')
        plt.title(f"Histograma: {col}", fontsize=10)
        plt.xlabel("")
        plt.ylabel("")
        
    plt.suptitle("Análise da Distribuição dos Sensores", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(figs_dir / "eda_histograms.png", dpi=200)
    plt.close()
    print(f"Histogramas salvos em: {figs_dir / 'eda_histograms.png'}")
    
    # C. Mapa de Calor de Correlação (Mantido do original)
    print("\n--- Gerando Mapa de Calor de Correlação ---")
    df_corr = df_train_filtered.drop(columns=['unit', 'cycle']).select_dtypes(include=np.number)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_corr.corr(), annot=False, cmap='viridis')
    plt.title("Matriz de Correlação das Features com RUL")
    plt.tight_layout()
    plt.savefig(figs_dir / "correlation_heatmap_enhanced.png", dpi=200)
    plt.close()
    print(f"Mapa de Calor salvo em: {figs_dir / 'correlation_heatmap_enhanced.png'}")


if __name__ == "__main__":
    main()