import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Carrega os dados processados."""
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
    """Prepara os dados para clustering."""
    # Filtra apenas reviews negativas (score <= 3)
    df_negative = df[df['review_score'] <= 3].copy()
    logger.info(f"Total de reviews negativas: {len(df_negative)}")
    return df_negative

def perform_clustering(df, n_clusters=5):
    """Realiza o clustering das reviews."""
    # Vetorização do texto
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['processed_text'])
    
    # Aplicação do K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    return df, vectorizer, kmeans

def visualize_clusters(df, vectorizer, kmeans):
    """Visualiza os clusters usando PCA."""
    # Redução de dimensionalidade para visualização
    X = vectorizer.transform(df['processed_text'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    
    # Criação do gráfico
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
    plt.title('Visualização dos Clusters de Reviews')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(scatter, label='Cluster')
    
    # Cria diretório de relatórios se não existir
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Salva o gráfico
    plt.savefig(reports_dir / 'cluster_visualization.png')
    plt.close()

def analyze_clusters(df):
    """Analisa e exibe informações sobre os clusters."""
    # Contagem de reviews por cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    logger.info("\nDistribuição de reviews por cluster:")
    logger.info(cluster_counts)
    
    # Exemplos de reviews por cluster
    logger.info("\nExemplos de reviews por cluster:")
    for cluster in range(len(cluster_counts)):
        examples = df[df['cluster'] == cluster]['review_text'].head(3)
        logger.info(f"\nCluster {cluster}:")
        for i, example in enumerate(examples, 1):
            logger.info(f"Exemplo {i}: {example}")

def save_models(vectorizer, kmeans, pca):
    """Salva os modelos treinados."""
    try:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(vectorizer, models_dir / 'cluster_vectorizer.joblib')
        joblib.dump(kmeans, models_dir / 'kmeans.joblib')
        joblib.dump(pca, models_dir / 'pca.joblib')
        logger.info("Modelos salvos com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar os modelos: {e}")

def main():
    logger.info("Iniciando análise de clustering...")
    
    # Carrega os dados
    df = load_data()
    if df is None:
        return
    
    # Prepara os dados
    df_negative = prepare_data(df)
    
    # Realiza o clustering
    df_clustered, vectorizer, kmeans = perform_clustering(df_negative)
    
    # Visualiza os clusters
    visualize_clusters(df_clustered, vectorizer, kmeans)
    
    # Analisa os clusters
    analyze_clusters(df_clustered)
    
    # Salva os modelos
    save_models(vectorizer, kmeans, PCA(n_components=2))

if __name__ == "__main__":
    main() 