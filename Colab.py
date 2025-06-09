# Instalação de dependências
!pip install deepface pandas pyarrow

# Importações necessárias
from google.colab import drive
import os
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
import numpy as np

# Configuração inicial
def setup_environment():
    """Configura o ambiente e caminhos"""
    # Monta o Google Drive
    drive.mount('/content/drive')
    
    # Define os caminhos
    dataset_path = '/content/drive/MyDrive/TestT1ED/Lfw_funneled'
    output_path = '/content/drive/MyDrive/TestT1ED/embeddings.parquet'
    
    # Verifica se o dataset existe
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado em {dataset_path}")
    
    return dataset_path, output_path

# Processamento do dataset
def process_dataset(dataset_path, max_images=None):
    """
    Processa o dataset e extrai informações das imagens
    
    Args:
        dataset_path: Caminho para o diretório do dataset
        max_images: Número máximo de imagens a processar (opcional)
    
    Returns:
        DataFrame com informações das imagens processadas
    """
    data = []
    extensions = {'.jpg', '.jpeg', '.png'}
    
    print(f"Processando dataset em {dataset_path}...")
    
    # Percorre todas as pastas de pessoas
    for person_name in tqdm(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        
        if os.path.isdir(person_dir):
            # Encontra a primeira imagem válida na pasta
            for file in os.listdir(person_dir):
                if os.path.splitext(file)[1].lower() in extensions:
                    img_path = os.path.join(person_dir, file)
                    data.append({
                        'person_name': person_name,
                        'image_path': img_path
                    })
                    break  # Usa apenas uma imagem por pessoa
        
        # Limita o número de imagens se especificado
        if max_images and len(data) >= max_images:
            break
    
    return pd.DataFrame(data)

# Extração de embeddings
def extract_embeddings(df, models=['Facenet', 'VGG-Face']):
    """
    Extrai embeddings faciais para cada imagem
    
    Args:
        df: DataFrame com informações das imagens
        models: Lista de modelos a tentar (em ordem de preferência)
    
    Returns:
        DataFrame com embeddings extraídos
    """
    embeddings = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        embedding = None
        
        # Tenta cada modelo até conseguir extrair o embedding
        for model in models:
            try:
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name=model,
                    detector_backend='retinaface',
                    enforce_detection=False
                )
                embedding = result[0]['embedding']
                break
            except Exception as e:
                continue
        
        embeddings.append(embedding if embedding else [None]*128)
    
    df['embedding'] = embeddings
    return df[df['embedding'].notna()]  # Remove falhas

# Fluxo principal
def main():
    # Configuração inicial
    dataset_path, output_path = setup_environment()
    
    # Processa o dataset (limitando a 1000 imagens para teste)
    print("\nProcessando imagens...")
    df = process_dataset(dataset_path, max_images=1000)
    
    # Extrai embeddings
    print("\nExtraindo embeddings faciais...")
    df_embeddings = extract_embeddings(df)
    
    # Converte embeddings para numpy array
    df_embeddings['embedding'] = df_embeddings['embedding'].apply(np.array)
    
    # Salva os resultados
    df_embeddings.to_parquet(output_path)
    print(f"\nEmbeddings salvos em {output_path}")
    
    # Estatísticas
    print("\nResumo do processamento:")
    print(f"- Total de faces processadas: {len(df_embeddings)}")
    print(f"- Dimensão dos embeddings: {len(df_embeddings['embedding'].iloc[0])}")

if __name__ == "__main__":
    main()
