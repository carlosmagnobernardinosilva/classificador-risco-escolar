"""
02_feature_engineering.py

Demonstra etapa de ENGENHARIA DE FEATURES:
- One-Hot Encoding para variáveis categóricas
- Padronização com Z-score (StandardScaler)
- Balanceamento de classes (Undersampling)

Execução:
    python 02_feature_engineering.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

from dados_sinteticos import gerar_dataset_sintetico


def aplicar_one_hot_encoding(df, colunas_categoricas):
    """
    Converte variáveis categóricas em numéricas usando One-Hot Encoding.
    
    Exemplo:
        'Fundamental' → [1, 0, 0, ...]
        'Médio'       → [0, 1, 0, ...]
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset com variáveis categóricas
    colunas_categoricas : list
        Nomes das colunas categóricas
        
    Returns:
    --------
    pd.DataFrame
        Dataset com variáveis categóricas convertidas
    """
    df = df.copy()
    
    print(f"\n🔀 ONE-HOT ENCODING:")
    print(f"   Colunas categóricas a processar: {colunas_categoricas}")
    
    # Aplicar get_dummies
    df_encoded = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)
    
    print(f"   Colunas antes: {df.shape[1]}")
    print(f"   Colunas depois: {df_encoded.shape[1]}")
    print(f"   ✅ One-Hot Encoding aplicado com sucesso")
    
    return df_encoded


def padronizar_features_numericas(df, colunas_numericas):
    """
    Padroniza (Z-score) variáveis numéricas.
    
    Fórmula: z = (x - μ) / σ
    
    Resultado:
    - Média = 0
    - Desvio padrão = 1
    - Range: ~[-3, 3]
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset com features numéricas
    colunas_numericas : list
        Nomes das colunas numéricas
        
    Returns:
    --------
    tuple: (df_padronizado, scaler)
        DataFrame com features padronizadas e objeto scaler
    """
    df = df.copy()
    
    print(f"\n📊 PADRONIZAÇÃO COM Z-SCORE:")
    print(f"   Colunas a padronizar: {colunas_numericas}")
    
    scaler = StandardScaler()
    df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
    
    print(f"\n   Estatísticas após padronização:")
    for col in colunas_numericas:
        print(f"   - {col}:")
        print(f"      Média: {df[col].mean():.6f}")
        print(f"      Desvio padrão: {df[col].std():.6f}")
    
    print(f"   ✅ Padronização aplicada com sucesso")
    
    return df, scaler


def diagnosticar_desbalanceamento(y):
    """
    Diagnostica o desbalanceamento de classes.
    
    Parameters:
    -----------
    y : pd.Series or np.array
        Variável alvo (0/1)
        
    Returns:
    --------
    dict
        Estatísticas sobre balanceamento
    """
    unique, counts = np.unique(y, return_counts=True)
    
    info = {
        'classe_0_count': counts[0] if 0 in unique else 0,
        'classe_1_count': counts[1] if 1 in unique else 0,
        'classe_0_pct': (counts[0] / len(y) * 100) if 0 in unique else 0,
        'classe_1_pct': (counts[1] / len(y) * 100) if 1 in unique else 0,
        'proporcao': max(counts) / min(counts)
    }
    
    return info


def balancear_classes_undersampling(X, y, random_state=42):
    """
    Aplica undersampling para balancear classes desproporcionais.
    
    Estratégia: Reduz a classe majoritária para igualar com a minoritária.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    random_state : int
        Seed para reprodutibilidade
        
    Returns:
    --------
    tuple: (X_balanced, y_balanced)
        Features e target balanceados
    """
    print(f"\n⚖️  BALANCEAMENTO DE CLASSES:")
    
    # Diagnóstico antes
    info_antes = diagnosticar_desbalanceamento(y)
    print(f"\n   ANTES DO BALANCEAMENTO:")
    print(f"   - Classe 0 (Aprovado): {info_antes['classe_0_count']} ({info_antes['classe_0_pct']:.1f}%)")
    print(f"   - Classe 1 (Reprovado): {info_antes['classe_1_count']} ({info_antes['classe_1_pct']:.1f}%)")
    print(f"   - Proporção: {info_antes['proporcao']:.1f}:1")
    
    # Aplicar undersampling
    undersampler = RandomUnderSampler(random_state=random_state)
    X_balanced, y_balanced = undersampler.fit_resample(X, y)
    
    # Diagnóstico depois
    info_depois = diagnosticar_desbalanceamento(y_balanced)
    print(f"\n   DEPOIS DO BALANCEAMENTO (Undersampling):")
    print(f"   - Classe 0 (Aprovado): {info_depois['classe_0_count']} ({info_depois['classe_0_pct']:.1f}%)")
    print(f"   - Classe 1 (Reprovado): {info_depois['classe_1_count']} ({info_depois['classe_1_pct']:.1f}%)")
    print(f"   - Proporção: {info_depois['proporcao']:.1f}:1 (BALANCEADO ✓)")
    
    print(f"\n   ✅ Undersampling aplicado com sucesso")
    
    return X_balanced, y_balanced


def gerar_relatorio_features(df, X_final, y):
    """
    Gera relatório final sobre as features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset original
    X_final : pd.DataFrame
        Features finais
    y : pd.Series
        Target
    """
    print(f"\n" + "="*60)
    print("RESUMO DA ENGENHARIA DE FEATURES")
    print("="*60)
    
    print(f"\n📊 DIMENSÕES:")
    print(f"   Features originais: {df.shape[1]}")
    print(f"   Features finais: {X_final.shape[1]}")
    print(f"   Amostras: {X_final.shape[0]}")
    
    print(f"\n📈 CLASSES:")
    print(f"   Distribuição:")
    print(f"   - Aprovados: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"   - Reprovados: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    print(f"\n📋 FEATURES FINAIS:")
    print(f"   {list(X_final.columns)}")
    
    print(f"\n📊 AMOSTRA DE DADOS:")
    print(X_final.head(5).to_string())
    
    print(f"\n✅ TRANSFORMAÇÕES APLICADAS:")
    print(f"   ✓ One-Hot Encoding (variáveis categóricas)")
    print(f"   ✓ Padronização Z-score (features numéricas)")
    print(f"   ✓ Balanceamento com Undersampling")
    
    print(f"\n" + "="*60 + "\n")


# ============================================================================
# MAIN - Execução Completa
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("ETAPA 2: ENGENHARIA DE FEATURES")
    print("="*60)
    
    # 1. Gerar dados (já pré-processados)
    print("\n📥 Carregando dados sintéticos...")
    df = gerar_dataset_sintetico(n_samples=300)
    
    # Simular pré-processamento básico (que seria feito antes)
    df = df.drop(['id_anonimizado'], axis=1)
    
    # 2. Separar features e target
    print("🔀 Separando features e target...")
    X = df.drop('reprovado', axis=1)
    y = df['reprovado']
    
    print(f"   Features: {X.shape[1]} colunas")
    print(f"   Target: Classe {y.unique()}")
    
    # 3. One-Hot Encoding
    X = aplicar_one_hot_encoding(X, ['serie', 'nivel_ensino'])
    
    # 4. Padronização
    X, scaler = padronizar_features_numericas(X, ['notas_finais', 'total_faltas_minutos'])
    
    # 5. Balanceamento
    X_balanced, y_balanced = balancear_classes_undersampling(X, y)
    
    # 6. Relatório
    gerar_relatorio_features(df, X_balanced, y_balanced)
    
    # 7. Salvar resultado
    X_balanced.to_csv('features_engenharia.csv', index=False)
    y_balanced.to_csv('target_balanceado.csv', index=False)
    print("✅ Features salvas em 'features_engenharia.csv'")
    print("✅ Target balanceado salvo em 'target_balanceado.csv'")
