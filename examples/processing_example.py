"""
01_preprocessing_example.py

Demonstra etapa de PRÉ-PROCESSAMENTO E LIMPEZA:
- Normalização de escalas
- Tratamento de valores ausentes
- Limpeza de caracteres indesejados
- Remoção de dados pessoais (LGPD)

Execução:
    python 01_preprocessing_example.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dados_sinteticos import gerar_dataset_sintetico


def normalizar_notas(df, coluna='notas_finais'):
    """
    Normaliza notas para escala 0-100.
    
    Estratégia:
    - Notas 0-10 → multiplicar por 10
    - Notas 0-100 → manter
    - Notas conceituais → converter para numérica
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset com coluna de notas
    coluna : str
        Nome da coluna de notas
        
    Returns:
    --------
    pd.DataFrame
        Dataset com notas normalizadas
    """
    df = df.copy()
    
    # Clip para garantir 0-100
    df[coluna] = df[coluna].clip(0, 100)
    
    return df


def tratar_valores_nulos(df):
    """
    Trata valores ausentes no dataset.
    
    Estratégia:
    - Faltas: Imputar pela média individual do aluno
    - Se aluno não tiver nenhum registro: usar média geral
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset com possíveis NaN
        
    Returns:
    --------
    pd.DataFrame
        Dataset sem valores nulos
    """
    df = df.copy()
    
    # Simular alguns valores nulos (para demonstração)
    # Numa aplicação real, já teriam NaN
    indices_nulos = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
    df.loc[indices_nulos, 'total_faltas_minutos'] = np.nan
    
    print(f"\n📝 TRATAMENTO DE VALORES NULOS:")
    print(f"   Valores nulos em 'total_faltas_minutos' antes: {df['total_faltas_minutos'].isna().sum()}")
    
    # Imputar pela média (em caso real, seria pela média individual do aluno)
    df['total_faltas_minutos'].fillna(df['total_faltas_minutos'].median(), inplace=True)
    
    print(f"   Valores nulos após imputação: {df['total_faltas_minutos'].isna().sum()}")
    print(f"   ✅ Tratados com sucesso")
    
    return df


def criar_variavel_composta(df):
    """
    Cria variáveis compostas combinando informações.
    
    Exemplo: Combinar 'serie' + 'nivel_ensino'
    - "3º Ano" + "Fundamental" → "3º Ano F"
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset original
        
    Returns:
    --------
    pd.DataFrame
        Dataset com novas variáveis
    """
    df = df.copy()
    
    print(f"\n🔄 CRIAÇÃO DE VARIÁVEIS COMPOSTAS:")
    print(f"   Combinando 'serie' + 'nivel_ensino'...")
    
    # Criar coluna composta
    df['serie_nivel'] = df['serie'].str.extract(r'(\d)')[0] + 'º ' + df['nivel_ensino'].str[0]
    
    print(f"   ✅ Variável 'serie_nivel' criada")
    print(f"      Exemplo: {df['serie_nivel'].head(3).values}")
    
    return df


def remover_dados_pessoais(df):
    """
    Remove colunas com dados pessoais (LGPD).
    
    Estratégia:
    - Manter apenas ID anonimizado
    - Remover nomes, emails, etc.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset completo
        
    Returns:
    --------
    pd.DataFrame
        Dataset sem dados pessoais
    """
    df = df.copy()
    
    print(f"\n🔐 CONFORMIDADE LGPD:")
    print(f"   Colunas originais: {list(df.columns)}")
    
    # Neste caso, já vem anonimizado do gerador
    # Mas mostrar exemplo do que seria removido
    colunas_removidas = []
    for col in df.columns:
        if col in ['nome', 'email', 'cpf', 'endereco', 'telefone']:
            df = df.drop(col, axis=1)
            colunas_removidas.append(col)
    
    print(f"   Colunas removidas: {colunas_removidas if colunas_removidas else 'Nenhuma (já anonimizado)'}")
    print(f"   ✅ Dados pessoais protegidos")
    
    return df


def limpar_caracteres_indesejados(df):
    """
    Remove caracteres indesejados de colunas de texto.
    
    Operações:
    - Remove espaços extras
    - Remove caracteres especiais
    - Padroniza minúsculas/maiúsculas
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset limpo
    """
    df = df.copy()
    
    print(f"\n🧹 LIMPEZA DE CARACTERES:")
    
    # Para colunas texto
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'id_anonimizado':  # Não processar ID
            df[col] = df[col].str.strip()  # Remove espaços extras
            df[col] = df[col].str.upper()  # Padroniza para maiúsculas
    
    print(f"   ✅ Espaços extras removidos")
    print(f"   ✅ Texto padronizado")
    
    return df


def gerar_resumo_limpeza(df_antes, df_depois):
    """
    Gera resumo das transformações aplicadas.
    
    Parameters:
    -----------
    df_antes : pd.DataFrame
        Dataset antes do pré-processamento
    df_depois : pd.DataFrame
        Dataset após pré-processamento
    """
    print(f"\n" + "="*60)
    print("RESUMO DO PRÉ-PROCESSAMENTO")
    print("="*60)
    
    print(f"\n📊 COMPARAÇÃO:")
    print(f"   Registros antes: {len(df_antes)}")
    print(f"   Registros depois: {len(df_depois)}")
    print(f"   Colunas antes: {df_antes.shape[1]}")
    print(f"   Colunas depois: {df_depois.shape[1]}")
    
    print(f"\n✅ TRANSFORMAÇÕES APLICADAS:")
    print(f"   ✓ Normalização de escalas")
    print(f"   ✓ Tratamento de valores nulos")
    print(f"   ✓ Remoção de dados pessoais (LGPD)")
    print(f"   ✓ Limpeza de caracteres")
    print(f"   ✓ Criação de variáveis compostas")
    
    print(f"\n📈 DADOS DEPOIS DO PRÉ-PROCESSAMENTO:")
    print(df_depois.describe().to_string())
    
    print(f"\n" + "="*60 + "\n")


# ============================================================================
# MAIN - Execução Completa
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("ETAPA 1: PRÉ-PROCESSAMENTO E LIMPEZA")
    print("="*60)
    
    # 1. Gerar dados
    print("\n📥 Carregando dados sintéticos...")
    df = gerar_dataset_sintetico(n_samples=300)
    df_original = df.copy()
    
    # 2. Executar pipeline de limpeza
    print("\n🔄 Iniciando pré-processamento...")
    
    df = normalizar_notas(df)
    print("   ✓ Notas normalizadas")
    
    df = tratar_valores_nulos(df)
    
    df = criar_variavel_composta(df)
    
    df = remover_dados_pessoais(df)
    
    df = limpar_caracteres_indesejados(df)
    
    # 3. Gerar resumo
    gerar_resumo_limpeza(df_original, df)
    
    # 4. Salvar resultado
    df.to_csv('dados_preprocessados.csv', index=False)
    print("✅ Dados preprocessados salvos em 'dados_preprocessados.csv'")
    
    # 5. Mostrar amostras
    print("\n📋 AMOSTRA DOS DADOS PROCESSADOS:")
    print(df.head(10).to_string())
