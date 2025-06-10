# ED-UFMS
Estruturas de Dados UFMS 
Trabalho 1 - # Análise e Documentação do Sistema de Reconhecimento Facial com KDTree

## 1. Visão Geral do Sistema

O sistema desenvolvido implementa um mecanismo de reconhecimento facial utilizando embeddings faciais armazenados em uma estrutura KDTree. A solução consiste em três componentes principais:

1. **Biblioteca C**: Implementa a KDTree otimizada para armazenar e buscar embeddings faciais
2. **API FastAPI**: Interface REST para interação com a KDTree
3. **Scripts de processamento**: Para extração de embeddings e teste do sistema

## 2. Alterações Implementadas na KDTree

### 2.1 Estrutura de Dados Modificada

A estrutura original para coordenadas geográficas foi substituída por uma estrutura adequada para embeddings faciais:

```c
// Estrutura original (coordenadas)
typedef struct _reg {
    int lat;
    int lon;
    char nome[100];
} treg;

// Nova estrutura (embeddings faciais)
typedef struct _reg {
    float embedding[128];  // Vetor de características da face
    char id[100];          // Identificador da pessoa
} treg;
```

### 2.2 Implementação do Heap para Busca de N Vizinhos

Foi adicionado um sistema de heap mínimo para retornar os N vizinhos mais próximos:

```c
typedef struct {
    double *distancias;
    treg **vizinhos;
    int capacidade;
    int tamanho;
} heap_vizinhos;

void heap_init(heap_vizinhos *heap, int capacidade);
void heap_push(heap_vizinhos *heap, double dist, treg *vizinho);
void _kdtree_busca_n(tarv *arv, tnode *atual, void *key, int profund, heap_vizinhos *heap);
```

## 3. Fluxo de Trabalho do Sistema

1. **Extrair embeddings faciais** (colab.py):
   - Processa imagens do dataset LFW
   - Usa DeepFace para gerar embeddings de 128 dimensões
   - Armazena em arquivo Parquet

2. **Construir a KDTree** (app.py):
   - Inicializa a estrutura da árvore
   - Configura funções de comparação e distância

3. **Inserir faces**:
   - Cada embedding é inserido na árvore com um ID associado

4. **Buscar faces similares**:
   - Dado um embedding de consulta, retorna os N vizinhos mais próximos

## 4. Testes e Validação

O sistema inclui três níveis de teste:

1. **Testes unitários em C** (kdtree.c):
   - Verificam construção da árvore
   - Validam operações de inserção e busca

2. **Testes de API** (test.py):
   - Verificam integração entre Python e C
   - Testam endpoints REST

3. **Validação com dataset real**:
   - Usa imagens do LFW dataset
   - Verifica reconhecimento correto de faces conhecidas

## 5. Exemplo de Uso

```python
# 1. Construir a árvore
response = requests.post("http://127.0.0.1:8000/construir-arvore")

# 2. Inserir faces
for nome, embedding in zip(nomes, embeddings):
    payload = {
        "embedding": embedding,
        "id": nome
    }
    requests.post("http://127.0.0.1:8000/inserir-face", json=payload)

# 3. Buscar faces similares
response = requests.post(
    "http://127.0.0.1:8000/buscar-faces",
    json=embedding_consulta,
    params={"n": 5}
)
```

## 6. Conclusão e Melhorias Futuras

O sistema foi refatorado com sucesso para:

- Suportar vetores de alta dimensionalidade (128D)
- Implementar busca eficiente de múltiplos vizinhos
- Manter boa performance mesmo com milhares de registros

**Melhorias potenciais**:

1. Adicionar persistência da árvore em disco
2. Implementar balanceamento automático
3. Adicionar suporte a atualização/remoção de registros
4. Melhorar tratamento de erros na API

O projeto demonstra como estruturas de dados especializadas (como a KDTree) podem ser adaptadas para aplicações modernas de visão computacional, mantendo eficiência mesmo com grandes volumes de dados.
