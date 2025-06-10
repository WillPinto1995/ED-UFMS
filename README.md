# ED-UFMS
Estruturas de Dados UFMS 
Trabalho 1 - Sistema de Reconhecimento Facial com KDTree
Visão Geral do Projeto
Este projeto implementa um sistema completo de reconhecimento facial utilizando:

DeepFace para extração de embeddings faciais

KDTree (Árvore k-dimensional) para armazenamento e busca eficiente

FastAPI para disponibilizar uma API REST

Google Colab para processamento inicial dos dados

Estrutura do Projeto
text
/projeto
│
├── app.py               # API FastAPI para reconhecimento facial
├── kdtree_wrapper.py    # Wrapper Python para a KDTree em C
├── kdtree.c             # Implementação da KDTree em C
├── colab.py             # Script para extração de embeddings no Colab
├── test.py              # Testes para a API
└── README.md            # Documentação do projeto
Componentes Principais
1. colab.py - Extração de Embeddings
Funcionalidades:

Processa o dataset LFW (Labeled Faces in the Wild)

Extrai embeddings faciais usando DeepFace

Armazena resultados em formato Parquet

Fluxo de Trabalho:

Monta o Google Drive

Acessa o dataset em /content/drive/MyDrive/TestT1ED/Lfw_funneled

Processa até 1000 imagens (1 por pessoa)

Extrai embeddings com modelos Facenet ou VGG-Face

Salva em /content/drive/MyDrive/TestT1ED/embeddings.parquet

Configurações:

Modelos alternativos em caso de falha

Limite configurável de imagens

Tratamento robusto de erros

2. kdtree.c - Implementação da KDTree
Estruturas de Dados:

c
typedef struct {
    float embedding[128];  // Vetor de características
    char id[100];          // Identificador
} FaceRegistro;

typedef struct TreeNode {
    void *key;            // Ponteiro para registro
    struct TreeNode *esq; // Subárvore esquerda
    struct TreeNode *dir; // Subárvore direita
} TreeNode;

typedef struct {
    TreeNode *raiz;       // Raiz da árvore
    int dimensoes;        // Número de dimensões (128)
    // Funções de comparação e distância
} KDTree;
Funcionalidades Implementadas:

Inserção balanceada de nós

Busca por vizinho mais próximo

Busca dos N vizinhos mais próximos (com heap)

Função de distância euclidiana quadrada

Otimizações:

Busca em O(log n) no caso médio

Heap para manutenção dos melhores resultados

Podas na busca para melhor performance

3. kdtree_wrapper.py - Interface Python/C
Componentes:

Definições de estruturas compatíveis com o C

Tipos de dados para a interface

Wrapper com métodos Pythonicos:

python
class KDTreeWrapper:
    def insert_face(self, embedding: list, face_id: str):
        """Insere uma face na árvore"""
    
    def search_faces(self, embedding: list, n: int = 5):
        """Busca n faces similares"""
4. app.py - API FastAPI
Endpoints:

POST /build-tree: Constrói a árvore vazia

POST /insert-face: Insere um novo embedding

POST /search-faces: Busca faces similares

Validações:

Embeddings devem ter 128 dimensões

IDs limitados a 100 caracteres

Tratamento de erros detalhado

Modelos Pydantic:

python
class FaceInput(BaseModel):
    embedding: List[float]
    id: str

class FaceSearchResult(BaseModel):
    id: str
    similarity: float
5. test.py - Testes da API
Casos de Teste:

Construção da árvore

Inserção de 100 faces de exemplo

Busca por similaridade

Funcionalidades:

Carga de embeddings do arquivo Parquet

Testes automatizados com tratamento de erros

Relatório de resultados

Como Executar o Projeto
Pré-requisitos
Python 3.8+

Google Colab (para extração inicial)

Biblioteca DeepFace

FastAPI

Pandas

ctypes

Passo a Passo
Extrair embeddings (Colab):

bash
python colab.py
Compilar a KDTree:

bash
gcc -c -fpic kdtree.c
gcc -shared -o libkdtree.so kdtree.o
Iniciar a API:

bash
uvicorn app:app --reload
Executar testes:

bash
python test.py
Exemplo de Uso
Inserir uma face:

python
import requests

data = {
    "embedding": [0.1, 0.2, ...], # 128 valores
    "id": "pessoa1"
}

response = requests.post("http://localhost:8000/insert-face", json=data)
Buscar faces similares:

python
response = requests.post(
    "http://localhost:8000/search-faces",
    json=[0.1, 0.2, ...], # Embedding de consulta
    params={"n": 5}
)
Considerações Finais
Desafios Resolvidos:

Integração Python/C com ctypes

Busca eficiente em alta dimensionalidade

Processamento de grandes volumes de imagens

Possíveis Melhorias:

Persistência da árvore em disco

Balanceamento automático

Suporte a atualização/remoção

API assíncrona

Este projeto demonstra uma aplicação prática de estruturas de dados avançadas (KDTree) combinada com técnicas modernas de visão computacional para resolver um problema real de reconhecimento facial de forma eficiente.
