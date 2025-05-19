import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Carrega e prepara os dados para treinamento."""
    try:
        data_dir = Path('data')
        input_file = data_dir / 'processed_reviews.csv'
        df = pd.read_csv(input_file)
        logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar os dados: {e}")
        return None

def prepare_data(df):
    """Prepara os dados para treinamento."""
    # Considera reviews com score <= 3 como negativas (0) e > 3 como positivas (1)
    df['sentiment'] = (df['review_score'] > 3).astype(int)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Treina o modelo de classificação."""
    # Vetorização do texto
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Treinamento do modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Avalia o modelo treinado."""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred)
    logger.info("\nRelatório de Classificação:")
    logger.info(report)
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("\nMatriz de Confusão:")
    logger.info(conf_matrix)

def save_model(model, vectorizer):
    """Salva o modelo e o vetorizador."""
    try:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / 'classifier.joblib'
        vectorizer_path = models_dir / 'vectorizer.joblib'
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Modelo salvo em: {model_path}")
        logger.info(f"Vetorizador salvo em: {vectorizer_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar o modelo: {e}")

def main():
    logger.info("Iniciando treinamento do modelo...")
    
    # Carrega os dados
    df = load_data()
    if df is None:
        return
    
    # Prepara os dados
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Treina o modelo
    model, vectorizer = train_model(X_train, y_train)
    
    # Avalia o modelo
    evaluate_model(model, vectorizer, X_test, y_test)
    
    # Salva o modelo
    save_model(model, vectorizer)

if __name__ == "__main__":
    main() 