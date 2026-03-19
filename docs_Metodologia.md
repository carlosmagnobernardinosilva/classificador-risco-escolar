# 📚 Metodologia Detalhada - Classificador de Risco de Reprovação Escolar

## 1. Introdução Metodológica

Este documento descreve em detalhes a abordagem cientifica e técnica empregada no desenvolvimento do classificador de diagnóstico para identificação de fatores de risco associados à reprovação escolar.

A metodologia segue o fluxo padrão de **Ciência de Dados**, dividido em 4 etapas principais:

```
1. PRÉ-PROCESSAMENTO → 2. ENGENHARIA DE FEATURES → 3. MODELAGEM → 4. AVALIAÇÃO
```

---

## 2. Fonte de Dados

### 2.1 Origem e Coleta

- **Instituição**: Escola da rede pública de Minas Gerais, Brasil
- **Período**: 2012-2024 (dados digitalizados)
- **Nível Educacional**: Ensino Fundamental I até Ensino Médio
- **Forma de Coleta**: Manual de registros escolares digitalizados
- **Conformidade**: Lei Geral de Proteção de Dados (LGPD - Lei nº 13.709/2018)

### 2.2 Características do Dataset

| Aspecto | Descrição |
|---------|-----------|
| **Total de Registros** | 1.141 registros (aluno × disciplina × ano) |
| **Período Temporal** | 12 anos de históricos |
| **Nível Educacional** | Fundamental I, II e Médio |
| **Variáveis Coletadas** | Notas, faltas, série, resultado final |
| **Taxa de Reprovação** | ~20% (desbalanceamento de classes) |

---

## 3. Fase 1: Pré-Processamento e Limpeza

### 3.1 Objetivos da Fase

- Garantir integridade dos dados
- Padronizar formatos heterogêneos
- Tratar valores ausentes
- Remover inconsistências e outliers
- Garantir conformidade LGPD

### 3.2 Etapas de Limpeza

#### 3.2.1 Normalização de Escalas de Notas

**Problema Inicial**: Notas em diferentes escalas
- Conceituais: "A" (100), "B" (75), "C" (60), "D" (50)
- Numéricas antigas: 0-10
- Numéricas modernas: 0-100
- Valores corrompidos em algumas linhas

**Solução Aplicada**:
```
Mapeamento Conceitual:
├── "A", "ÓTIMO", "MB" → 100
├── "B" → 75
├── "C" → 60
└── "D", "INSUFICIENTE" → 50

Normalização Numéricas:
├── Escala 0-10 → 0-100 (multiplicar por 10)
└── Escala 0-100 → mantém
```

**Resultado**: Todas as notas em escala 0-100

#### 3.2.2 Tratamento de Valores Ausentes

| Coluna | Problema | % NaN | Solução |
|--------|----------|-------|---------|
| `Aproveitamento` | Registros faltantes | ~3% | Imputação pela média individual |
| `Status_Final` | Dados ausentes | ~6% | Substituir por 9999 (marcador) |

**Estratégia de Imputação**: Média individual do aluno (não global)
- ✅ Preserva variabilidade individual
- ✅ Melhor que média geral ou mediana
- ✅ Alinhado com literatura (Han et al., 2011)

#### 3.2.3 Remoção de Dados Pessoais (LGPD)

Conformidade com Lei nº 13.709/2018:

```python
# Removido:
- Nome do aluno
- Número de matrícula
- CPF (se existisse)
- Data de nascimento
- Endereço

# Mantido:
- ID anonimizado (0, 1, 2, ...)
- Série/Ano escolar
- Notas por disciplina
- Faltas
- Situação final
```

#### 3.2.4 Limpeza de Caracteres Indesejados

- Remoção de espaços extras
- Padronização de acentuação
- Remoção de caracteres especiais
- Conversão para lowercase (quando aplicável)

---

## 4. Fase 2: Engenharia de Atributos (Feature Engineering)

### 4.1 Criação de Novas Variáveis

#### 4.1.1 Variável Composta: `Serie_Nivel`

**Motivação**: Diferenciar anos escolares homônimos

```
Fundamental:    3º Ano F
Médio:          3º Ano M
```

**Implementação**:
```python
df['Serie_Nivel'] = df['Serie'] + '_' + df['Nivel_Ensino']
```

**Benefício**: Permite capturar padrões específicos por nível educacional

#### 4.1.2 Variável Alvo: `Target` (Aprovado/Reprovado)

**Baseado em Norma Brasileira**:
- Score ≥ 60.0 → "Aprovado" (0)
- Score < 60.0 → "Reprovado" (1)

**Critério de Reprovação Anual**:
- Um aluno é marcado como "Reprovado no Ano" se:
  - Reprovou em **QUALQUER** disciplina daquele ano

```python
# Lógica:
aluno_reprovado_ano = any(nota_disciplina < 60 for all_disciplinas)
```

### 4.2 Transformações Numéricas

#### 4.2.1 One-Hot Encoding

Variáveis categóricas convertidas para numéricas:

```
Serie_Nivel:
├── 3º Fund → [1, 0, 0, 0, 0, 0, ...]
├── 5º Fund → [0, 1, 0, 0, 0, 0, ...]
└── ... (6 valores possíveis)
```

**Técnica**: `pd.get_dummies()` (sklearn)
**Resultado**: 6 novas colunas binárias

#### 4.2.2 Padronização com Z-Score (StandardScaler)

Para features numéricas: `Aproveitamento` e `Total_Faltas_Minutos`

```
Fórmula: z = (x - μ) / σ

Onde:
- x = valor original
- μ = média da feature
- σ = desvio padrão
```

**Resultado**:
- Média = 0
- Desvio padrão = 1
- Range: aproximadamente [-3, 3]

**Benefício**:
- Todos os modelos convergem melhor
- XGBoost não é sensível, mas Random Forest é
- Regressão Logística requer

---

## 5. Fase 3: Tratamento do Desbalanceamento de Classes

### 5.1 Problema: Classes Desproporcionais

```
Aprovados: 80% (classe majoritária)
Reprovados: 20% (classe minoritária)
```

**Impacto**: Modelos tendiam a prever sempre "Aprovado"

### 5.2 Estratégias Testadas

#### 5.2.1 Class Weight Balancing

```python
sklearn.ensemble.RandomForestClassifier(
    class_weight='balanced'
)
```

**Como funciona**: Aumenta peso dos reprovados no loss function

**Resultado**: AUC 0.849 (Random Forest), 0.966 (XGBoost)

#### 5.2.2 SMOTE (Synthetic Minority Over-Sampling)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Como funciona**: Cria amostras sintéticas da classe minoritária

**Resultado**: AUC 0.839 (Random Forest), 0.958 (XGBoost)

#### 5.2.3 Undersampling (VENCEDOR ⭐)

```python
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = undersampler.fit_resample(X, y)
```

**Como funciona**: Remove amostras da classe majoritária

**Antes**: 80% Aprovados, 20% Reprovados
**Depois**: 50% Aprovados, 50% Reprovados

**Resultado**: AUC 0.840 (Random Forest), **0.959 (XGBoost)** ✅

**Desvantagem**: Perde alguns dados de aprovados (aceitável)

### 5.3 Comparação de Técnicas

| Técnica | Random Forest AUC | XGBoost AUC | Vencedor |
|---------|-------------------|------------|----------|
| Class Weight | 0.849 | 0.966 | Class Weight (XGB) |
| SMOTE | 0.839 | 0.958 | Undersampling |
| **Undersampling** | **0.840** | **0.959** | ⭐ Escolhido |

---

## 6. Fase 4: Seleção e Treinamento de Modelos

### 6.1 Critérios de Seleção

Testamos 3 algoritmos com diferentes características:

| Modelo | Tipo | Complexidade | Vantagem |
|--------|------|-------------|----------|
| Regressão Logística | Linear | Baixa | Baseline, interpretável |
| Random Forest | Ensemble | Média | Não-linear, robusto |
| XGBoost | Ensemble Boosting | Alta | SOTA em tabular |

### 6.2 Configuração dos Hiperparâmetros

#### 6.2.1 Regressão Logística

```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Considerado em validação cruzada
)
```

#### 6.2.2 Random Forest

```python
RandomForestClassifier(
    n_estimators=100,      # 100 árvores
    max_depth=10,          # Profundidade máxima
    random_state=42,
    n_jobs=-1,             # Usar todos os cores
    class_weight='balanced'
)
```

#### 6.2.3 XGBoost (Melhor)

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,           # Árvores rasas
    learning_rate=0.1,     # Taxa de aprendizado
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
```

### 6.3 Validação Cruzada

**Estratégia**: k-fold com k=5

```
Dataset → Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5
           ├─ 80% train, 20% test
           ├─ 80% train, 20% test
           ├─ 80% train, 20% test
           ├─ 80% train, 20% test
           └─ 80% train, 20% test

Score final = Média das 5 rodadas
```

**Repetições**: 5 rodadas completas para robustez

**Métricas coletadas**:
- AUC (Área Sob a Curva ROC)
- Recall (Sensibilidade)
- Precisão
- F1-Score

---

## 7. Fase 5: Avaliação e Interpretação

### 7.1 Métricas Utilizadas

#### 7.1.1 AUC (Área Sob a Curva ROC)

**Fórmula**: Integral sob a curva ROC de 0 a 1

```
AUC = 1.0 → Perfeito
AUC = 0.9+ → Excelente
AUC = 0.8+ → Muito Bom
AUC = 0.7+ → Bom
AUC < 0.5 → Pior que aleatório
```

**Nosso Resultado**: 0.959 ✅ Excelente

#### 7.1.2 Recall (Revocação/Sensibilidade)

```
Recall = TP / (TP + FN)

Onde:
- TP (Verdadeiro Positivo) = Reprovado predito como reprovado
- FN (Falso Negativo) = Reprovado predito como aprovado

Significado: De todos os alunos que REALMENTE reprovaram,
             quantos meu modelo conseguiu identificar?
```

**Nosso Resultado**: 0.897 (89.7%)
- De 100 alunos reprovados, conseguimos identificar 89-90

#### 7.1.3 Precisão (Precision)

```
Precisão = TP / (TP + FP)

Onde:
- TP = Reprovado predito como reprovado
- FP = Aprovado predito como reprovado (alarme falso)

Significado: Quando meu modelo prevê "reprovado",
             em quantos % está correto?
```

**Nosso Resultado**: 0.855 (85.5%)
- De 100 alertas emitidos, 85 estão corretos

#### 7.1.4 F1-Score

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)

Equilíbrio entre Precision e Recall
Range: 0 a 1 (1 = perfeito)
```

**Nosso Resultado**: 0.875 ✅ Muito Bom

### 7.2 Trade-off: Recall vs Precisão

```
Recall Alto (0.897):
✅ Detecta quase todos os em risco
❌ Pode haver muitos falsos positivos

Precisão Alta (0.855):
✅ Poucos alarmes falsos
❌ Alguns alunos em risco não detectados

Nosso Balanço: 89.7% vs 85.5% → ÓTIMO
```

## 8. Matriz de Confusão (Interpretação Detalhada)

<img width="630" height="500" alt="image" src="https://github.com/user-attachments/assets/2fa234bc-3166-43df-ae01-6dc43f289614" />

### Análise Célula por Célula

**Verdadeiros Positivos (TP) = 411**
- Alunos corretamente identificados como reprovados
- Taxa de acurácia para esta classe: 411/471 = 87.3%

**Verdadeiros Negativos (TN) = 602**
- Alunos corretamente identificados como aprovados
- Taxa de acurácia para esta classe: 602/670 = 89.9%

**Falsos Negativos (FN) = 60**
- Alunos reprovados que modelo previu como aprovados
- Taxa de falha na detecção: 60/471 = 12.7%
- ⚠️ CRÍTICO: Estes alunos não receberão intervenção

**Falsos Positivos (FP) = 68**
- Alunos aprovados que modelo previu como reprovados
- Taxa de alarme falso: 68/670 = 10.1%
- ℹ️ ACEITÁVEL: Intervenção desnecessária, sem danos

### Recomendação Operacional

FN é mais custoso que FP em contexto educacional:
- FN: Aluno em risco não recebe ajuda → evade/reprova
- FP: Aluno aprovado recebe ajuda extra → sem danos

**Ajuste Recomendado**: Reduzir threshold de decisão para aumentar Recall

---

## 9. Curva ROC - Interpretação

A Curva ROC plota:
- **Eixo X**: Taxa de Falsos Positivos (1 - Especificidade)
- **Eixo Y**: Taxa de Verdadeiros Positivos (Recall)
<img width="888" height="642" alt="image" src="https://github.com/user-attachments/assets/fd4f9aa0-3cc6-421c-b318-fff06730f7a4" />

```
- Curva mais próxima do canto superior esquerdo = melhor
- AUC 0.959 significa: 95.9% de probabilidade modelo acertar em dois exemplos aleatórios

---
```
## 10. Workflow Resumido em Pseudocódigo

```
FUNÇÃO pipeline_classificacao():
    # 1. Carregar dados
    df = carregar_dados_escola()
    
    # 2. Pré-processamento
    df = normalizar_notas(df)
    df = tratar_valores_nulos(df)
    df = remover_dados_pessoais(df)
    
    # 3. Engenharia de features
    df['serie_nivel'] = combinar_serie_nivel(df)
    X = df[features]
    y = df['reprovado']
    
    # 4. Balanceamento
    undersampler = RandomUnderSampler()
    X_balanced, y_balanced = undersampler.fit_resample(X, y)
    
    # 5. Divisão train/test
    X_train, X_test, y_train, y_test = split_80_20(X_balanced, y_balanced)
    
    # 6. Treinamento
    PARA cada modelo IN [LogReg, RandomForest, XGBoost]:
        PARA cada fold IN [1 a 5]:
            modelo.fit(X_train_fold, y_train_fold)
            metricas = avaliar(modelo, X_test_fold, y_test_fold)
            salvar(metricas)
    
    # 7. Seleção do melhor
    melhor_modelo = MAX(metricas.auc)  # XGBoost com AUC 0.959
    
    # 8. Retorno
    RETORNE melhor_modelo, metricas
```

---

## 11. Suposições e Limitações Metodológicas

### 11.1 Suposições

1. **Dados Representativos**: Dados de 2012-2024 representam padrão geral
2. **Notas Precisas**: Todos os registros estão corretos
3. **Relação Linear**: Notas e faltas correlacionam com reprovação
4. **Estacionariedade**: Padrões não mudam drasticamente ao longo do tempo

### 11.2 Limitações

1. **Dataset Único**: Apenas uma escola (não generalizável)
2. **Features Limitadas**: Apenas notas e faltas (faltam contexto social/familiar)
3. **Possível Overfitting**: AUC 0.959 pode ser inflado
4. **Dados Históricos**: Padrões podem ter mudado pós-pandemia

---

## 12. Próximas Etapas Metodológicas Recomendadas

1. **Validação Externa**: Testar em outras escolas
2. **Mais Features**: Incluir variáveis socioeconômicas
3. **Análise de Fairness**: Detectar bias por gênero/raça
4. **Interpretabilidade**: Usar SHAP values
5. **Deployment**: Criar API REST ou Dashboard

---

## 📚 Referências Metodológicas

- Han, J., Kamber, M., & Pei, J. (2011). Data mining: concepts and techniques
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE TKDE
- Fávero, L. P., & Belfiore, P. (2024). Manual de análise de dados
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system

---

**Última atualização**: Março 2025
**Versão**: 1.0
**Status**: ✅ Completo
