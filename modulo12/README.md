# Sistema de Análise e Resposta Automática de Reclamações

Este projeto implementa um sistema inteligente para análise e resposta automática de reclamações de e-commerce, utilizando técnicas de Machine Learning e Processamento de Linguagem Natural.

## Funcionalidades

- Classificação automática de reclamações
- Agrupamento de reclamações similares
- Geração de respostas automáticas personalizadas
- Recuperação aumentada (RAG) de informações relevantes

## Estrutura do Projeto

```
modulo12/
├── data/                   # Diretório para datasets
├── src/                    # Código fonte
│   ├── preprocessing/      # Scripts de pré-processamento
│   ├── classification/     # Modelos de classificação
│   ├── clustering/         # Modelos de clustering
│   └── generation/         # Geração de respostas
├── notebooks/             # Jupyter notebooks para análise
├── requirements.txt       # Dependências do projeto
└── README.md             # Este arquivo
```

## Configuração

1. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
- Crie um arquivo `.env` na raiz do projeto
- Adicione suas chaves de API necessárias

## Uso

1. Pré-processamento dos dados:
```bash
python src/preprocessing/preprocess.py
```

2. Treinamento do modelo de classificação:
```bash
python src/classification/train.py
```

3. Execução do sistema completo:
```bash
python src/main.py
``` 