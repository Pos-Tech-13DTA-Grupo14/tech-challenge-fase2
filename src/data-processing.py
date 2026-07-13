import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_clean_data(filepath):
    """Carrega o dataset, remove a coluna 'Id', cria a variável alvo e trata valores nulos."""
    df = pd.read_csv(filepath)
    
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)
    
    if "quality" in df.columns:
        df["high_quality"] = (df["quality"] >= 7).astype(int)

    # Tratamento de valores nulos (mediana)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    return df

def engineer_features(df):
    """Cria as 4 novas features físico-químicas."""
    df_clean = df.copy()
    df_clean['so2_ratio'] = df_clean['free sulfur dioxide'] / (df_clean['total sulfur dioxide'] + 1e-6)
    df_clean['total_acidity'] = df_clean['fixed acidity'] + df_clean['volatile acidity']
    df_clean['alcohol_density_ratio'] = df_clean['alcohol'] / df_clean['density']
    df_clean['sugar_alcohol_ratio'] = df_clean['residual sugar'] / (df_clean['alcohol'] + 1e-6)
    return df_clean

def prepare_and_save_pipeline(df_clean, test_size=0.2, random_state=42):
    """Separa em treino/teste, padroniza, aplica SMOTE e salva os artefatos."""
    # Define features (todas exceto quality e high_quality)
    features = [col for col in df_clean.columns if col not in ['quality', 'high_quality']]
    
    X = df_clean[features]
    y = df_clean['high_quality']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Padronização
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=features)

    # SMOTE (apenas no treino)
    smote = SMOTE(random_state=random_state)
    X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)

    # Salva arquivos processados na pasta data/processed/
    os.makedirs('data/processed', exist_ok=True)
    pd.DataFrame(X_train_final, columns=features).to_csv('data/processed/X_train.csv', index=False)
    X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train_final, name='high_quality').to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    df_clean.to_csv('data/processed/wine_clean.csv', index=False)
    joblib.dump(scaler, 'data/processed/scaler.pkl')

    return X_train_final, X_test_scaled, y_train_final, y_test