"""
03_model_training.py

Demonstra etapa de TREINAMENTO DE MODELOS:
- Regressão Logística (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting) - MELHOR

Execução:
    python 03_model_training.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from dados_sinteticos import gerar_dataset_sintetico


def preparar_dados(df, random_state=42):
    """
    Prepara dados para modelagem:
    - Pré-processamento
    - Feature engineering
    - Balanceamento
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset bruto
    random_state : int
        Seed para reprodutibilidade
        
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test)
        Conjuntos de treinamento e teste
    """
    print("\n" + "="*60)
    print("PREPARAÇÃO DE DADOS")
    print("="*60)
    
    # 1. Remover ID
    df = df.drop('id_anonimizado', axis=1)
    
    # 2. Separar features e target
    X = df.drop('reprovado', axis=1)
    y = df['reprovado']
    
    print(f"\n📊 FEATURES E TARGET:")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # 3. One-Hot Encoding
    X = pd.get_dummies(X, columns=['serie', 'nivel_ensino'], drop_first=True)
    
    # 4. Padronização
    scaler = StandardScaler()
    X[['notas_finais', 'total_faltas_minutos']] = scaler.fit_transform(
        X[['notas_finais', 'total_faltas_minutos']]
    )
    
    # 5. Balanceamento
    undersampler = RandomUnderSampler(random_state=random_state)
    X_balanced, y_balanced = undersampler.fit_resample(X, y)
    
    print(f"\n⚖️  BALANCEAMENTO:")
    print(f"   Antes: {y.value_counts().to_dict()}")
    print(f"   Depois: {pd.Series(y_balanced).value_counts().to_dict()}")
    
    # 6. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=random_state, stratify=y_balanced
    )
    
    print(f"\n📚 SPLIT TRAIN/TEST:")
    print(f"   Training: {X_train.shape[0]} amostras")
    print(f"   Testing: {X_test.shape[0]} amostras")
    
    return X_train, X_test, y_train, y_test


def treinar_regressao_logistica(X_train, y_train):
    """
    Treina Regressão Logística (modelo baseline).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de treinamento
    y_train : pd.Series
        Target de treinamento
        
    Returns:
    --------
    LogisticRegression
        Modelo treinado
    """
    print("\n🤖 TREINANDO REGRESSÃO LOGÍSTICA...")
    
    modelo = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(X_train, y_train)
    
    print("   ✅ Regressão Logística treinada")
    
    return modelo


def treinar_random_forest(X_train, y_train):
    """
    Treina Random Forest (ensemble).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de treinamento
    y_train : pd.Series
        Target de treinamento
        
    Returns:
    --------
    RandomForestClassifier
        Modelo treinado
    """
    print("\n🤖 TREINANDO RANDOM FOREST...")
    
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    modelo.fit(X_train, y_train)
    
    print("   ✅ Random Forest treinado")
    
    return modelo


def treinar_xgboost(X_train, y_train):
    """
    Treina XGBoost (gradient boosting) - MELHOR MODELO.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de treinamento
    y_train : pd.Series
        Target de treinamento
        
    Returns:
    --------
    XGBClassifier
        Modelo treinado
    """
    print("\n🤖 TREINANDO XGBOOST...")
    
    modelo = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    modelo.fit(X_train, y_train)
    
    print("   ✅ XGBoost treinado")
    
    return modelo


def validacao_cruzada(modelo, X, y, cv=5, nome_modelo="Modelo"):
    """
    Executa validação cruzada com 5 folds.
    
    Parameters:
    -----------
    modelo : estimator
        Modelo sklearn
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Número de folds
    nome_modelo : str
        Nome para exibição
        
    Returns:
    --------
    list
        Scores de cada fold
    """
    scores = cross_val_score(modelo, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"\n📊 VALIDAÇÃO CRUZADA ({cv}-fold) - {nome_modelo}:")
    for i, score in enumerate(scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    print(f"   Média: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return scores


def comparar_modelos(modelos_dict, X_train, y_train):
    """
    Executa validação cruzada para todos os modelos.
    
    Parameters:
    -----------
    modelos_dict : dict
        Dicionário {nome: modelo}
    X_train : pd.DataFrame
        Features de treinamento
    y_train : pd.Series
        Target de treinamento
        
    Returns:
    --------
    dict
        Resultados de validação cruzada
    """
    print("\n" + "="*60)
    print("VALIDAÇÃO CRUZADA DE TODOS OS MODELOS")
    print("="*60)
    
    resultados = {}
    
    for nome, modelo in modelos_dict.items():
        scores = validacao_cruzada(modelo, X_train, y_train, cv=5, nome_modelo=nome)
        resultados[nome] = {
            'modelo': modelo,
            'scores': scores,
            'media': scores.mean(),
            'std': scores.std()
        }
    
    return resultados


def exibir_ranking(resultados):
    """
    Exibe ranking dos modelos.
    
    Parameters:
    -----------
    resultados : dict
        Resultados da validação cruzada
    """
    print("\n" + "="*60)
    print("RANKING DE MODELOS (AUC)")
    print("="*60)
    
    # Ordenar por média
    ranking = sorted(resultados.items(), key=lambda x: x[1]['media'], reverse=True)
    
    for posicao, (nome, info) in enumerate(ranking, 1):
        medal = "🥇" if posicao == 1 else "🥈" if posicao == 2 else "🥉"
        print(f"\n{medal} {posicao}º lugar: {nome}")
        print(f"   AUC: {info['media']:.4f} ± {info['std']:.4f}")
        print(f"   Intervalo: [{info['media']-info['std']:.4f}, {info['media']+info['std']:.4f}]")
    
    print("\n" + "="*60)
    
    melhor = ranking[0]
    print(f"\n🏆 MELHOR MODELO: {melhor[0]} com AUC {melhor[1]['media']:.4f}")


# ============================================================================
# MAIN - Execução Completa
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("ETAPA 3: TREINAMENTO DE MODELOS")
    print("="*60)
    
    # 1. Gerar e preparar dados
    print("\n📥 Gerando dados sintéticos...")
    df = gerar_dataset_sintetico(n_samples=500)
    
    X_train, X_test, y_train, y_test = preparar_dados(df)
    
    # 2. Treinar modelos
    print("\n" + "="*60)
    print("TREINAMENTO")
    print("="*60)
    
    modelo_lr = treinar_regressao_logistica(X_train, y_train)
    modelo_rf = treinar_random_forest(X_train, y_train)
    modelo_xgb = treinar_xgboost(X_train, y_train)
    
    # 3. Comparar com validação cruzada
    modelos = {
        'Regressão Logística': modelo_lr,
        'Random Forest': modelo_rf,
        'XGBoost': modelo_xgb
    }
    
    resultados = comparar_modelos(modelos, X_train, y_train)
    
    # 4. Ranking
    exibir_ranking(resultados)
    
    # 5. Salvar modelos (opcional)
    import joblib
    joblib.dump(modelo_xgb, 'modelo_xgboost.pkl')
    print("\n✅ Modelo XGBoost salvo em 'modelo_xgboost.pkl'")
