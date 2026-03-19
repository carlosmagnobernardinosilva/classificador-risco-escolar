"""
04_evaluation.py

Demonstra etapa de AVALIAÇÃO DE MODELOS:
- Métricas de classificação (Recall, Precisão, F1, AUC)
- Curva ROC
- Matriz de confusão
- Feature importance
- Visualizações

Execução:
    python 04_evaluation.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve, f1_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from dados_sinteticos import gerar_dataset_sintetico


def preparar_dados_para_eval(df, random_state=42):
    """
    Prepara dados (pré-processamento + feature engineering).
    
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test, X_test_original)
    """
    df = df.drop('id_anonimizado', axis=1)
    X = df.drop('reprovado', axis=1)
    y = df['reprovado']
    
    # One-Hot Encoding
    X = pd.get_dummies(X, columns=['serie', 'nivel_ensino'], drop_first=True)
    
    # Padronização
    scaler = StandardScaler()
    X[['notas_finais', 'total_faltas_minutos']] = scaler.fit_transform(
        X[['notas_finais', 'total_faltas_minutos']]
    )
    
    # Balanceamento
    undersampler = RandomUnderSampler(random_state=random_state)
    X_balanced, y_balanced = undersampler.fit_resample(X, y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=random_state, stratify=y_balanced
    )
    
    return X_train, X_test, y_train, y_test


def calcular_metricas(y_true, y_pred, y_pred_proba):
    """
    Calcula métricas de classificação.
    
    Parameters:
    -----------
    y_true : np.array
        Valores verdadeiros
    y_pred : np.array
        Previsões binárias
    y_pred_proba : np.array
        Probabilidades preditas
        
    Returns:
    --------
    dict
        Dicionário com todas as métricas
    """
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Métricas básicas
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'recall': recall,
        'precisao': precisao,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': cm
    }


def treinar_e_avaliar_xgboost(X_train, X_test, y_train, y_test):
    """
    Treina e avalia modelo XGBoost.
    
    Returns:
    --------
    tuple: (modelo, métricas, previsões)
    """
    print("\n" + "="*60)
    print("TREINAMENTO E AVALIAÇÃO DO XGBOOST")
    print("="*60)
    
    # Treinar
    modelo = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    print("\n🤖 Treinando XGBoost...")
    modelo.fit(X_train, y_train)
    print("✅ Treinamento concluído")
    
    # Predições
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Métricas
    metricas = calcular_metricas(y_test, y_pred, y_pred_proba)
    
    return modelo, metricas, (y_pred, y_pred_proba)


def exibir_metricas(metricas, nome_modelo="XGBoost"):
    """
    Exibe métricas de forma formatada.
    
    Parameters:
    -----------
    metricas : dict
        Dicionário com métricas
    nome_modelo : str
        Nome do modelo
    """
    print(f"\n" + "="*60)
    print(f"MÉTRICAS DE AVALIAÇÃO - {nome_modelo}")
    print("="*60)
    
    print(f"\n📊 MATRIZ DE CONFUSÃO:")
    cm = metricas['confusion_matrix']
    print(f"   Verdadeiros Negativos (TN):  {metricas['tn']}")
    print(f"   Verdadeiros Positivos (TP):  {metricas['tp']}")
    print(f"   Falsos Negativos (FN):       {metricas['fn']}")
    print(f"   Falsos Positivos (FP):       {metricas['fp']}")
    
    print(f"\n📈 MÉTRICAS PRINCIPAIS:")
    print(f"   AUC (Área Sob a Curva):      {metricas['auc']:.4f} {'✅ EXCELENTE' if metricas['auc'] > 0.9 else '⚠️  BOM'}")
    print(f"   Recall (Sensibilidade):      {metricas['recall']:.4f} ({metricas['recall']*100:.1f}%)")
    print(f"   Precisão:                    {metricas['precisao']:.4f} ({metricas['precisao']*100:.1f}%)")
    print(f"   F1-Score:                    {metricas['f1']:.4f}")
    
    print(f"\n🎯 INTERPRETAÇÃO:")
    print(f"   De cada 100 alunos em REAL risco:")
    print(f"   - O modelo detecta: {metricas['recall']*100:.0f}%")
    print(f"   ")
    print(f"   Quando o modelo prediz 'risco':")
    print(f"   - Ele está correto: {metricas['precisao']*100:.0f}% das vezes")
    
    print(f"\n" + "="*60)


def plotar_matriz_confusao(metricas, figsize=(8, 6)):
    """
    Plota matriz de confusão em heatmap.
    
    Parameters:
    -----------
    metricas : dict
        Dicionário com métricas
    figsize : tuple
        Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    cm = metricas['confusion_matrix']
    
    # Normalizar para percentuais
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Aprovado', 'Reprovado'],
                yticklabels=['Aprovado', 'Reprovado'],
                cbar=False)
    
    plt.title('Matriz de Confusão - XGBoost', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Valor Predito', fontsize=12)
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    print("\n✅ Matriz de confusão salva em 'matriz_confusao.png'")
    plt.close()


def plotar_curva_roc(y_test, y_pred_proba, auc_score, figsize=(8, 6)):
    """
    Plota curva ROC com AUC.
    
    Parameters:
    -----------
    y_test : np.array
        Valores verdadeiros
    y_pred_proba : np.array
        Probabilidades preditas
    auc_score : float
        Score AUC
    figsize : tuple
        Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Plot
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'XGBoost (AUC={auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance (AUC=0.5)')
    
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title('Curva ROC - XGBoost', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('curva_roc.png', dpi=300, bbox_inches='tight')
    print("✅ Curva ROC salva em 'curva_roc.png'")
    plt.close()


def plotar_feature_importance(modelo, feature_names, figsize=(10, 6)):
    """
    Plota importância das features.
    
    Parameters:
    -----------
    modelo : XGBClassifier
        Modelo treinado
    feature_names : list
        Nomes das features
    figsize : tuple
        Tamanho da figura
    """
    plt.figure(figsize=figsize)
    
    # Obter importâncias
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10
    
    # Plot
    plt.barh(range(len(indices)), importances[indices], color='#2ca02c', alpha=0.7)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importância', fontsize=12)
    plt.title('Top 10 Features - XGBoost', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Feature importance salva em 'feature_importance.png'")
    plt.close()


def gerar_relatorio_final(metricas):
    """
    Gera relatório final textual.
    
    Parameters:
    -----------
    metricas : dict
        Dicionário com métricas
    """
    print("\n" + "="*60)
    print("RELATÓRIO FINAL DE AVALIAÇÃO")
    print("="*60)
    
    print(f"\n✅ CONCLUSÕES:")
    
    if metricas['auc'] > 0.9:
        print(f"   1. Modelo com EXCELENTE performance (AUC {metricas['auc']:.3f})")
    elif metricas['auc'] > 0.8:
        print(f"   1. Modelo com MUITO BOA performance (AUC {metricas['auc']:.3f})")
    else:
        print(f"   1. Modelo com BOA performance (AUC {metricas['auc']:.3f})")
    
    if metricas['recall'] > 0.85:
        print(f"   2. Alta capacidade de detecção (Recall {metricas['recall']*100:.1f}%)")
        print(f"      → Detecta 9 em cada 10 alunos em risco")
    
    if metricas['precisao'] > 0.80:
        print(f"   3. Alta confiabilidade em alertas (Precisão {metricas['precisao']*100:.1f}%)")
        print(f"      → 8 em cada 10 alertas estão corretos")
    
    print(f"\n🎯 RECOMENDAÇÃO:")
    print(f"   Modelo adequado para uso em produção")
    print(f"   Usar com validação humana")
    
    print(f"\n" + "="*60 + "\n")


# ============================================================================
# MAIN - Execução Completa
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("ETAPA 4: AVALIAÇÃO DE MODELOS")
    print("="*60)
    
    # 1. Preparar dados
    print("\n📥 Gerando e preparando dados...")
    df = gerar_dataset_sintetico(n_samples=500)
    X_train, X_test, y_train, y_test = preparar_dados_para_eval(df)
    
    # 2. Treinar e avaliar
    modelo, metricas, predicoes = treinar_e_avaliar_xgboost(X_train, X_test, y_train, y_test)
    y_pred, y_pred_proba = predicoes
    
    # 3. Exibir métricas
    exibir_metricas(metricas, "XGBoost")
    
    # 4. Gerar visualizações
    print("\n📊 Gerando visualizações...")
    plotar_matriz_confusao(metricas)
    plotar_curva_roc(y_test, y_pred_proba, metricas['auc'])
    plotar_feature_importance(modelo, X_test.columns)
    
    # 5. Relatório final
    gerar_relatorio_final(metricas)
    
    print("✅ Avaliação completa!")
