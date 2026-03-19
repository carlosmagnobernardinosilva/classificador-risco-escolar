# 📚 Classificador de Risco de Reprovação Escolar com Machine Learning

> Desenvolvimento de um classificador baseado em XGBoost para identificação precoce de fatores de risco associados à reprovação escolar em instituições de ensino público

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-green?style=flat-square)

---

## 📖 Sobre o Projeto

Este repositório contém a documentação, metodologia e insights principais de um **Trabalho de Conclusão de Curso** em Data Science Analytics, focado no desenvolvimento de um sistema de diagnóstico baseado em machine learning.

**Objetivo Principal**: Identificar padrões e variáveis preditoras de reprovação escolar em instituições de ensino público, permitindo intervenções pedagógicas preventivas e direcionadas.

### 📊 Características do Estudo

- **Instituição**: Escola da rede pública de Minas Gerais, Brasil
- **Período de Dados**: 2012-2024 (históricos escolares digitalizados)
- **Nível de Cobertura**: Ensino Fundamental I até Ensino Médio
- **Conformidade**: ✅ LGPD (Lei Geral de Proteção de Dados Pessoais)
- **Algoritmo Principal**: XGBoost com undersampling para balanceamento de classes

---

## 🎯 Resultados Principais

### Performance do Modelo XGBoost

| Métrica | Valor | Interpretação |
|---------|-------|----------------|
| **AUC (Área Sob a Curva)** | **0.959** | Excelente discriminação entre classes |
| **Recall (Sensibilidade)** | **0.897** | Detecta ~90% dos alunos em risco |
| **Precisão** | **0.855** | 85.5% de confiabilidade nos alertas |
| **F1-Score** | **0.875** | Equilíbrio ótimo entre precision/recall |

### Fatores Determinantes Identificados

🔴 **Variáveis Mais Importantes**:
1. **Desempenho Acadêmico (Notas Finais)** — Fator crítico
2. **Absenteísmo (Total de Faltas)** — Indicador forte de risco
3. **Série/Ano Escolar** — Variação por nível de ensino

> **Insight**: Modelos de ensemble (XGBoost) superaram significativamente regressão logística, demonstrando relacionamentos não-lineares nos dados

---

## 🏗️ Arquitetura da Solução

### 1️⃣ Pré-Processamento e Limpeza
- ✅ Padronização de escalas de notas (conceitual → numérica 0-100)
- ✅ Tratamento de valores ausentes (imputação por média individual)
- ✅ Remoção de dados pessoais (conformidade LGPD)
- ✅ Remoção de caracteres indesejados e normalização

### 2️⃣ Engenharia de Atributos
```
Transformações Realizadas:
├── Criação de variável composta (Serie_Nivel)
├── Construção da variável alvo (Target: Aprovado/Reprovado)
├── One-Hot Encoding para variáveis categóricas
├── Padronização com Z-score (StandardScaler)
└── Balanceamento de classes (undersampling)
```

### 3️⃣ Desenvolvimento do Classificador
**Modelos Testados**:
- Regressão Logística (baseline)
- Random Forest (ensemble)
- **XGBoost** ⭐ (campeão)

**Técnicas de Balanceamento**:
- Class Weight Adjustment
- SMOTE (Synthetic Minority Oversampling)
- **Undersampling** ⭐ (melhor performance)

### 4️⃣ Validação e Avaliação
- Validação Cruzada k-fold (5 rodadas)
- Matriz de Confusão detalhada
- Curva ROC com AUC
- Métricas de classificação binária

---

## 📊 Descobertas Principais

### ✅ Insight #1: Forte Correlação Desempenho-Frequência
- Alunos aprovados: Média de faltas **significativamente menor**
- Alunos reprovados: Alta variabilidade em absenteísmo
- **Conclusão**: Frequência é fator distintivo para reprovação

### ✅ Insight #2: Absenteísmo Elevado = Risco Crítico
- Distribuição de faltas é **assimétrica (right-skewed)**
- Pequeno grupo com ausência crítica (<5% dos alunos)
- **Implicação**: Esses alunos requerem intervenção imediata

### ✅ Insight #3: Superioridade de Modelos de Ensemble
```
Random Forest:      AUC 0.840 ✓
XGBoost:            AUC 0.959 ⭐⭐⭐
Regressão Logística: AUC 0.712 ✗
```
- XGBoost captura padrões não-lineares
- Adequado para relacionamentos complexos em dados educacionais

### ✅ Insight #4: Balanceamento de Classes é Crítico
**Técnica Escolhida**: Undersampling
- Recall 0.897 vs 0.631 (com class_weight)
- Recall 0.897 vs 0.678 (com SMOTE)
- **Resultado**: Melhor detecção de alunos em risco

---

## 🛠️ Tecnologias Utilizadas

| Ferramenta | Versão | Propósito |
|-----------|--------|----------|
| **Python** | 3.9+ | Linguagem principal |
| **Pandas** | Latest | Manipulação e limpeza de dados |
| **NumPy** | Latest | Cálculos numéricos |
| **Scikit-learn** | Latest | Modelos ML, preprocessamento |
| **XGBoost** | Latest | Gradient boosting |
| **Imbalanced-learn** | Latest | Técnicas de balanceamento |
| **Matplotlib** | Latest | Visualizações estáticas |
| **Seaborn** | Latest | Gráficos estatísticos |
| **Jupyter Notebook** | Latest | Análise interativa |

---

## 📁 Estrutura do Repositório

```
classificador-risco-escolar/
│
├── README.md                          
├── [TCC] - Carlos Magno Bernardino da Silva #TCC completo
├── docs
│   ├── Metodologia.md                 # Descrição detalhada da metodologia
│   ├── Resultados_Principais.md       # Tabelas e gráficos de resultados
│
├── examples/
│   ├── exemplo_preprocessamento.py    # Exemplo de pré-processamento
│   ├── exemplo_treinamento.py         # Exemplo de treinamento do modelo
│   └── exemplo_predicao.py            # Exemplo de uso em produção
```

---

## 🚀 Como Usar Este Repositório

### 1. Instalação de Dependências

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/school-risk-classifier.git
cd school-risk-classifier

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Explorar a Documentação

```bash
# Leitura recomendada:
1. README.md (este arquivo)
2. docs/Metodologia.md (entender o processo)
3. docs/Resultados_Principais.md (visualizar resultados)
4. TECHNICAL_DOCUMENTATION.md (detalhes técnicos)
```

### 3. Revisar Exemplos de Código

```bash
# Exemplos práticos de implementação
python examples/exemplo_preprocessamento.py
python examples/exemplo_treinamento.py
python examples/exemplo_predicao.py
```

### 4. Executar Testes

```bash
# Validar integridade do código
pytest tests/
```

---

## ⚠️ Limitações e Considerações

### 🔴 Limitações Identificadas

1. **Escopo Único**: Dados de uma única instituição
   - Modelo foi otimizado para padrões dessa escola
   - Generalização limitada para outras instituições
   - **Recomendação**: Validar em múltiplas escolas

2. **Variáveis Limitadas**: Apenas 2 features principais (notas + faltas)
   - Não inclui fatores socioeconômicos
   - Não inclui avaliações comportamentais
   - Não inclui contexto familiar
   - **Risco**: Possível overfitting

3. **Possível Overfitting**: AUC 0.959 é muito alto
   - Com apenas 2 variáveis, separação clara entre classes
   - **Mitigação**: Validação cruzada reduz este risco

---

## 📊 Comparação de Modelos

### Performance Completa (Validação Cruzada 5-fold)

```
┌──────────────────┬────────────┬──────────┬──────────┬────────┐
│ Modelo           │ Recall ↑   │ Precisão │ F1-Score │ AUC    │
├──────────────────┼────────────┼──────────┼──────────┼────────┤
│ Logistic Reg.    │ 0.626      │ 0.565    │ 0.593    │ 0.712  │
│ Random Forest    │ 0.716      │ 0.696    │ 0.705    │ 0.840  │
│ XGBoost ⭐      │ 0.897      │ 0.855    │ 0.875    │ 0.959  │
└──────────────────┴────────────┴──────────┴──────────┴────────┘
```

**Técnica de Balanceamento**: Undersampling aplicado em todos os modelos

---

## 🔐 Conformidade com LGPD

Este projeto **respeita integralmente a Lei Geral de Proteção de Dados Pessoais**:

✅ Dados pessoais removidos (nomes, matrícula, etc.)
✅ Identificadores anonimizados (ID numérico)
✅ Dados sensíveis não compartilhados no GitHub
✅ Notebooks com dados não são públicos
✅ Documentação preserva privacidade dos alunos

---

## 📝 Metodologia Resumida

### Fase 1: Exploração (EDA)
- Distribuição de notas: Concentrada 60-100
- Distribuição de faltas: Assimétrica (right-skewed)
- Clear separation entre aprovados/reprovados

### Fase 2: Preparação
- Limpeza de 100% dos dados
- Padronização de escalas
- Tratamento de valores ausentes
- Engenharia de features

### Fase 3: Modelagem
- Comparação 3 algoritmos
- Teste 3 técnicas de balanceamento
- Validação cruzada k-fold
- Seleção de hiperparâmetros

### Fase 4: Avaliação
- Métricas de classificação
- Curva ROC e AUC
- Matriz de confusão
- Análise de importância de features

---


## 📚 Referências Acadêmicas

- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE TKDE.
- Rumberger, R. W., & Lim, S. (2008). Why students drop out of school.
- LGPD - Lei nº 13.709/2018 (Brasil)


