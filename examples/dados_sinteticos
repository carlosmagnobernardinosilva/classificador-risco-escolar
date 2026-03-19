"""
dados_sinteticos.py

Gerador de dados SINTÉTICOS para demonstração.
Cria dataset similar ao original mas com dados ficcionais.
Não contém informações reais de alunos (LGPD Compliant).

Uso:
    from dados_sinteticos import gerar_dataset_sintetico
    df = gerar_dataset_sintetico(n_samples=500)
"""

import pandas as pd
import numpy as np


def gerar_dataset_sintetico(n_samples=500, random_state=42):
    """
    Gera dataset SINTÉTICO com características similares aos dados reais.
    
    Estrutura imitada do dataset original:
    - Notas finais (0-100)
    - Total de faltas (em minutos)
    - Série/Nível escolar
    - Situação final (Aprovado/Reprovado)
    
    Parameters:
    -----------
    n_samples : int
        Número de registros a gerar (default: 500)
    random_state : int
        Seed para reprodutibilidade
        
    Returns:
    --------
    pd.DataFrame
        Dataset com colunas: [id, notas_finais, total_faltas, serie, reprovado]
        
    Examples:
    ---------
    >>> df = gerar_dataset_sintetico(n_samples=100)
    >>> print(df.shape)
    (100, 5)
    >>> print(df.head())
    """
    np.random.seed(random_state)
    
    # 1. Gerar Notas Finais
    # Real: concentrado 60-100 com pico próximo 100
    # 80% aprovados (notas > 60), 20% reprovados (notas < 60)
    notas_aprovados = np.random.normal(loc=78, scale=12, 
                                       size=int(n_samples * 0.8)).clip(60, 100)
    notas_reprovados = np.random.normal(loc=35, scale=15, 
                                        size=int(n_samples * 0.2)).clip(0, 59)
    notas_finais = np.concatenate([notas_aprovados, notas_reprovados])
    
    # 2. Gerar Faltas (em minutos)
    # Real: distribuição assimétrica (right-skewed)
    # Maioria com baixa ausência, cauda longa com alta ausência
    faltas_baixas = np.random.exponential(scale=1200, 
                                          size=int(n_samples * 0.85))  # 85% baixa ausência
    faltas_altas = np.random.exponential(scale=5000, 
                                         size=int(n_samples * 0.15))   # 15% alta ausência
    total_faltas = np.concatenate([faltas_baixas, faltas_altas])
    
    # 3. Gerar Série/Nível Escolar
    series_possiveis = [
        '3º Ano F', '5º Ano F', '8º Ano F',  # Fundamental
        '1º Ano M', '2º Ano M', '3º Ano M'   # Médio
    ]
    serie = np.random.choice(series_possiveis, size=n_samples)
    
    # 4. Extrair Nível de Ensino
    nivel = np.where(serie.str.contains('F'), 'Fundamental', 'Médio')
    
    # 5. Criar Variável Alvo: Reprovado (1) ou Aprovado (0)
    # Se nota < 60 → Reprovado
    reprovado = (notas_finais < 60).astype(int)
    
    # 6. Montar DataFrame
    df = pd.DataFrame({
        'id_anonimizado': range(1, n_samples + 1),
        'notas_finais': notas_finais,
        'total_faltas_minutos': total_faltas.astype(int),
        'serie': serie,
        'nivel_ensino': nivel,
        'reprovado': reprovado
    })
    
    # 7. Embaralhar
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def estatisticas_dataset(df):
    """
    Retorna estatísticas básicas do dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset de entrada
        
    Returns:
    --------
    dict
        Dicionário com estatísticas
    """
    stats = {
        'total_registros': len(df),
        'total_features': df.shape[1],
        'alunos_unicos': df['id_anonimizado'].nunique(),
        'taxa_reprovacao': df['reprovado'].mean() * 100,
        'media_notas': df['notas_finais'].mean(),
        'mediana_notas': df['notas_finais'].median(),
        'media_faltas': df['total_faltas_minutos'].mean(),
        'mediana_faltas': df['total_faltas_minutos'].median(),
        'series_unicas': df['serie'].nunique(),
    }
    return stats


def imprimir_info_dataset(df):
    """
    Imprime informações resumidas do dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset para análise
    """
    print("\n" + "="*60)
    print("INFORMAÇÕES DO DATASET SINTÉTICO")
    print("="*60)
    
    stats = estatisticas_dataset(df)
    
    print(f"\n📊 TAMANHO:")
    print(f"   Total de registros: {stats['total_registros']}")
    print(f"   Total de features: {stats['total_features']}")
    print(f"   Alunos únicos: {stats['alunos_unicos']}")
    
    print(f"\n📈 DESEMPENHO ESCOLAR:")
    print(f"   Taxa de reprovação: {stats['taxa_reprovacao']:.1f}%")
    print(f"   Média de notas: {stats['media_notas']:.1f}")
    print(f"   Mediana de notas: {stats['mediana_notas']:.1f}")
    
    print(f"\n📝 ABSENTEÍSMO:")
    print(f"   Média de faltas: {stats['media_faltas']:.0f} minutos ({stats['media_faltas']/60:.1f} horas)")
    print(f"   Mediana de faltas: {stats['mediana_faltas']:.0f} minutos ({stats['mediana_faltas']/60:.1f} horas)")
    
    print(f"\n🏫 ESTRUTURA:")
    print(f"   Séries/Anos únicos: {stats['series_unicas']}")
    print(f"   {df['nivel_ensino'].value_counts().to_dict()}")
    
    print(f"\n✅ PRIMEIRAS LINHAS:")
    print(df.head(10).to_string())
    print("\n" + "="*60 + "\n")


# ============================================================================
# MAIN - Para Usar como Script Standalone
# ============================================================================

if __name__ == '__main__':
    print("\n🔄 Gerando dataset sintético...")
    
    # Gerar dados
    df = gerar_dataset_sintetico(n_samples=500)
    
    # Exibir informações
    imprimir_info_dataset(df)
    
    # Salvar (opcional)
    df.to_csv('dados_sinteticos_escola.csv', index=False)
    print("✅ Dataset salvo em 'dados_sinteticos_escola.csv'")
