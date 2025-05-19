import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import openai

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()

class ComplaintAnalyzer:
    def __init__(self):
        """Inicializa o analisador de reclamações."""
        self.classifier = None
        self.vectorizer = None
        self.kmeans = None
        self.cluster_vectorizer = None
        self.pca = None
        
        # Configuração da API OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Carrega os modelos
        self.load_models()
    
    def load_models(self):
        """Carrega os modelos treinados."""
        try:
            self.classifier = joblib.load('models/classifier.joblib')
            self.vectorizer = joblib.load('models/vectorizer.joblib')
            self.kmeans = joblib.load('models/kmeans.joblib')
            self.cluster_vectorizer = joblib.load('models/cluster_vectorizer.joblib')
            self.pca = joblib.load('models/pca.joblib')
            logger.info("Modelos carregados com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar os modelos: {e}")
    
    def classify_complaint(self, text):
        """Classifica uma reclamação como positiva ou negativa."""
        if not self.classifier or not self.vectorizer:
            raise ValueError("Modelos não carregados")
        
        # Pré-processa o texto
        processed_text = self.preprocess_text(text)
        
        # Vetoriza o texto
        X = self.vectorizer.transform([processed_text])
        
        # Faz a predição
        prediction = self.classifier.predict(X)[0]
        probability = self.classifier.predict_proba(X)[0]
        
        return {
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(max(probability))
        }
    
    def cluster_complaint(self, text):
        """Agrupa a reclamação em um cluster."""
        if not self.kmeans or not self.cluster_vectorizer:
            raise ValueError("Modelos de clustering não carregados")
        
        # Pré-processa o texto
        processed_text = self.preprocess_text(text)
        
        # Vetoriza o texto
        X = self.cluster_vectorizer.transform([processed_text])
        
        # Faz a predição do cluster
        cluster = self.kmeans.predict(X)[0]
        
        return {
            'cluster': int(cluster)
        }
    
    def generate_response(self, text, classification, cluster):
        """Gera uma resposta automática para a reclamação."""
        prompt = f"""
        Reclamação: {text}
        Classificação: {classification['sentiment']} (confiança: {classification['confidence']:.2f})
        Cluster: {cluster['cluster']}
        
        Gere uma resposta profissional e empática para esta reclamação, considerando:
        1. O sentimento da reclamação
        2. O cluster ao qual pertence
        3. A necessidade de resolver o problema de forma eficiente
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente de atendimento ao cliente profissional e empático."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return "Desculpe, não foi possível gerar uma resposta no momento."
    
    def preprocess_text(self, text):
        """Pré-processa o texto para análise."""
        # Implementação simplificada do pré-processamento
        text = text.lower()
        text = ' '.join(text.split())
        return text

def main():
    # Cria diretórios necessários
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    
    # Inicializa o analisador
    analyzer = ComplaintAnalyzer()
    
    # Exemplo de uso
    example_complaint = """
    O produto chegou com defeito e o atendimento foi péssimo.
    Estou muito insatisfeito com a demora na resposta.
    """
    
    try:
        # Classifica a reclamação
        classification = analyzer.classify_complaint(example_complaint)
        logger.info(f"Classificação: {classification}")
        
        # Agrupa a reclamação
        cluster = analyzer.cluster_complaint(example_complaint)
        logger.info(f"Cluster: {cluster}")
        
        # Gera resposta
        response = analyzer.generate_response(example_complaint, classification, cluster)
        logger.info(f"\nResposta gerada:\n{response}")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {e}")

if __name__ == "__main__":
    main() 