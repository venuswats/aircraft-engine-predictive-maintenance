from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib # Usado para salvar o scaler
from utils import feature_columns, set_reproducible

def load_and_prepare_data():
    ROOT = Path(__file__).resolve().parent.parent
    train_prep_path = ROOT / "outputs" / "tables" / "train_prepared.csv"
    test_prep_path = ROOT / "outputs" / "tables" / "test_prepared.csv"
    
    df_train = pd.read_csv(train_prep_path)
    df_test = pd.read_csv(test_prep_path)

    # 1. REMOÇÃO DE FEATURES CONSTANTES (REPETIR ETAPA DO EDA)
    # Lista padrão para C-MAPSS (Ajuste com base na saída do seu 00_eda.py)
    CONSTANT_COLS = ["setting3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"] 
    df_train = df_train.drop(columns=CONSTANT_COLS, errors='ignore')
    df_test = df_test.drop(columns=CONSTANT_COLS, errors='ignore')

    # Definir Features (X) e Target (y)
    FEATURES = feature_columns(df_train)
    TARGET = 'RUL'

    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    X_test = df_test[FEATURES]
    
    return X_train, y_train, X_test, FEATURES, TARGET, ROOT

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, features: list, root: Path):
    """Aplica o MinMaxScaler e salva o objeto scaler."""
    # 2. ESCALONAMENTO DE FEATURES (CRÍTICO para MLPs)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 
    
    # Converter de volta para DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

    # Salvar o objeto scaler (necessário para inferência)
    scaler_path = root / "outputs" / "models" / "mlp_scaler.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvo em {scaler_path}")
    
    return X_train_scaled, X_test_scaled, scaler_path

def create_mlp_model(input_dim):
    """Cria o modelo MLP com arquitetura definida."""
    model = Sequential([
        # Camada de Entrada + 1ª Oculta
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2), # Ajuda a prevenir overfitting
        # 2ª Camada Oculta
        Dense(64, activation='relu'),
        Dropout(0.2), 
        # Camada de Saída (Regressão da RUL)
        Dense(1, activation='linear')
    ])
    
    # Compilação conforme o TCC: Otimizador Adam e função de perda MSE
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def main():
    set_reproducible(42)
    X_train, y_train, X_test, FEATURES, TARGET, ROOT = load_and_prepare_data()

    X_train_scaled, X_test_scaled, _ = scale_data(X_train, X_test, FEATURES, ROOT)

    # 3. Divisão de Treino/Validação (para Early Stopping)
    X_t, X_val, y_t, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # 4. Construção e Resumo do Modelo
    model = create_mlp_model(input_dim=X_t.shape[1])
    model.summary()
    
    model_output_dir = ROOT / "outputs" / "models"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Callbacks para Treinamento Robusto
    callbacks = [
        # EarlyStopping: Monitora a perda de validação, para se não melhorar após 10 épocas
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
        # ModelCheckpoint: Salva o modelo que obteve a melhor performance na validação
        ModelCheckpoint(
            str(model_output_dir / "best_mlp_model.keras"), 
            monitor='val_loss', save_best_only=True, mode='min', verbose=1
        )
    ]

    # 6. Treinamento
    history = model.fit(
        X_t, y_t,
        epochs=100,
        batch_size=256, # Batch size maior para treinamento mais rápido
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    print("Treinamento concluído. O melhor modelo foi salvo automaticamente pelo ModelCheckpoint.")

if __name__ == "__main__":
    main()