🎯 Sobre o Projeto
Este projeto tem como objetivo desenvolver uma pipeline completa de Machine Learning para prever a qualidade sensorial de vinhos tintos através de dados físico-químicos.

A avaliação tradicional é um processo oneroso e subjetivo, dependente de especialistas. Transformamos essa "arte" em ciência, utilizando o Wine Quality Dataset para classificar vinhos entre Alta Qualidade (nota ≥ 7) e Baixa/Média Qualidade (nota < 7), auxiliando na tomada de decisão baseada em dados.

🚀 Etapas do Projeto
Seguimos a metodologia exigida para garantir uma análise robusta e reprodutível:

Compreensão do Problema: Definição do objetivo, tratamento da variável alvo e binarização.

Análise Exploratória (EDA): Investigação de distribuições, correlações e detecção de outliers.

Pré-processamento: Engenharia de features, padronização (StandardScaler) e tratamento do desbalanceamento de classes (SMOTE).

Modelagem: Desenvolvimento e ajuste de hiperparâmetros (GridSearchCV) utilizando Random Forest.

Avaliação e Interpretação: Análise de métricas (F1-Score, Matriz de Confusão) e extração da importância das variáveis para o negócio.

📂 Estrutura do Projeto
Plaintext
/
├── data/
│   ├── raw/           # Dataset original
│   └── processed/     # Dados tratados e balanceados (SMOTE)
├── notebooks/         # Jupyter Notebooks de desenvolvimento
├── results/           # Gráficos, métricas e modelos salvos (.pkl)
├── src/               # Scripts auxiliares e código-fonte
├── requirements.txt   # Dependências do projeto
└── README.md          # Documentação do projeto
🛠️ Tecnologias Utilizadas
Linguagem: Python 3.x

Manipulação de Dados: Pandas, NumPy

Modelagem e ML: Scikit-learn, Imbalanced-learn (SMOTE)

Visualização: Matplotlib, Seaborn

Ambiente: VS Code / Jupyter Notebook / Google Colab

💻 Como executar
Para configurar o ambiente localmente (recomendado para o desenvolvimento no VS Code):

Clone o repositório:

Bash
git clone https://github.com/Pos-Tech-13DTA-Grupo14/tech-challenge-fase2
Crie e ative um ambiente virtual:

Bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
Instale as dependências:

Bash
pip install -r requirements.txt
👥 Equipe
Grupo 14: Lucas Nunes Melaré Coelho, Gabriel, Sara, André e Raul.

Este projeto faz parte da grade curricular da pós-graduação em Data Analytics - FIAP (Fase 2).
