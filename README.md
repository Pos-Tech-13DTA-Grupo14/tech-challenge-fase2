# 🍷 Tech Challenge - Fase 2 | Análise de Qualidade de Vinhos

**Pós-Graduação em Data Analytics - FIAP**  
**Turma:** 13DTA | **Grupo:** 14

---

## 🎯 Sobre o Projeto
Este projeto tem como objetivo desenvolver uma pipeline completa de *Machine Learning* para prever a qualidade sensorial de vinhos tintos através de dados físico-químicos. 

A avaliação tradicional é um processo oneroso e subjetivo, dependente de especialistas. Transformamos essa "arte" em ciência, utilizando o *Wine Quality Dataset* para classificar vinhos entre **Alta Qualidade (nota ≥ 7)** e **Baixa/Média Qualidade (nota < 7)**, auxiliando na tomada de decisão baseada em dados.

---

## 🚀 Etapas do Projeto
Seguimos a metodologia exigida para garantir uma análise robusta e reprodutível:

1. **Compreensão do Problema:** Definição do objetivo, tratamento da variável alvo e binarização.
2. **Análise Exploratória (EDA):** Investigação de distribuições, correlações e detecção de *outliers*.
3. **Pré-processamento:** Engenharia de *features*, padronização (*StandardScaler*) e tratamento do desbalanceamento de classes (*SMOTE*).
4. **Modelagem:** Desenvolvimento e ajuste de hiperparâmetros (*GridSearchCV*) utilizando *Random Forest*.
5. **Avaliação e Interpretação:** Análise de métricas (*F1-Score, Matriz de Confusão*) e extração da importância das variáveis para o negócio.

---

## 📂 Estrutura do Projeto

```text
/
├── data/
│   ├── raw/           # Dataset original
│   └── processed/     # Dados tratados e balanceados (SMOTE)
├── notebooks/         # Jupyter Notebooks de desenvolvimento
├── results/           # Gráficos, métricas e modelos salvos (.pkl)
├── src/               # Scripts auxiliares e código-fonte
├── requirements.txt   # Dependências do projeto
└── README.md          # Documentação do projeto
