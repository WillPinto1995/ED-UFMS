#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <assert.h>

// ==================== ESTRUTURAS DE DADOS ====================

typedef struct {
    float embedding[128];  // Vetor de características da face
    char id[100];          // Identificador da pessoa (99 chars + null terminator)
} FaceRegistro;

typedef struct TreeNode {
    void *key;                  // Ponteiro para FaceRegistro
    struct TreeNode *esq;       // Subárvore esquerda
    struct TreeNode *dir;       // Subárvore direita
} TreeNode;

typedef struct {
    TreeNode *raiz;             // Raiz da árvore
    int (*comparador)(void*, void*, int);  // Função de comparação
    double (*distancia)(void*, void*);     // Função de distância
    int dimensoes;             // Número de dimensões (k)
} KDTree;

// ==================== FUNÇÕES AUXILIARES ====================

// Aloca e inicializa um novo registro de face
FaceRegistro* criar_registro(float embedding[128], const char id[]) {
    FaceRegistro *reg = malloc(sizeof(FaceRegistro));
    if (!reg) return NULL;
    
    memcpy(reg->embedding, embedding, sizeof(float) * 128);
    strncpy(reg->id, id, 99);
    reg->id[99] = '\0';
    
    return reg;
}

// Compara dois registros em uma dimensão específica
int comparar_faces(void *a, void *b, int dimensao) {
    float diff = ((FaceRegistro*)a)->embedding[dimensao] - ((FaceRegistro*)b)->embedding[dimensao];
    return (diff < 0) ? -1 : (diff > 0) ? 1 : 0;
}

// Calcula distância euclidiana quadrada entre duas faces
double distancia_faces(void *a, void *b) {
    double soma = 0.0;
    for (int i = 0; i < 128; i++) {
        double diff = ((FaceRegistro*)a)->embedding[i] - ((FaceRegistro*)b)->embedding[i];
        soma += diff * diff;
    }
    return soma;
}

// ==================== OPERAÇÕES DA KD-TREE ====================

// Inicializa uma nova KD-Tree
void kdtree_inicializar(KDTree *arv, int dimensoes) {
    arv->raiz = NULL;
    arv->comparador = comparar_faces;
    arv->distancia = distancia_faces;
    arv->dimensoes = dimensoes;
}

// Função auxiliar recursiva para inserção
void _kdtree_inserir(TreeNode **no, void *chave, int profundidade, KDTree *arv) {
    if (*no == NULL) {
        *no = malloc(sizeof(TreeNode));
        (*no)->key = chave;
        (*no)->esq = NULL;
        (*no)->dir = NULL;
    } else {
        int dim = profundidade % arv->dimensoes;
        if (arv->comparador((*no)->key, chave, dim) < 0) {
            _kdtree_inserir(&(*no)->dir, chave, profundidade + 1, arv);
        } else {
            _kdtree_inserir(&(*no)->esq, chave, profundidade + 1, arv);
        }
    }
}

// Insere um novo registro na árvore
void kdtree_inserir(KDTree *arv, void *chave) {
    _kdtree_inserir(&arv->raiz, chave, 0, arv);
}

// ==================== BUSCA DE VIZINHOS ====================

// Estrutura para armazenar os N vizinhos mais próximos
typedef struct {
    double *distancias;    // Array de distâncias
    FaceRegistro **faces;  // Array de ponteiros para registros
    int capacidade;        // Número máximo de vizinhos
    int tamanho;           // Número atual de vizinhos
} HeapVizinhos;

// Inicializa o heap de vizinhos
void heap_inicializar(HeapVizinhos *heap, int capacidade) {
    heap->distancias = malloc(sizeof(double) * capacidade);
    heap->faces = malloc(sizeof(FaceRegistro*) * capacidade);
    heap->capacidade = capacidade;
    heap->tamanho = 0;
}

// Adiciona um vizinho ao heap (mantém os mais próximos)
void heap_adicionar_vizinho(HeapVizinhos *heap, double dist, FaceRegistro *face) {
    if (heap->tamanho < heap->capacidade) {
        // Heap não está cheio, simplesmente adiciona
        heap->distancias[heap->tamanho] = dist;
        heap->faces[heap->tamanho] = face;
        heap->tamanho++;
        
        // Reorganiza o heap
        int i = heap->tamanho - 1;
        while (i > 0 && heap->distancias[i] > heap->distancias[(i-1)/2]) {
            // Troca com o pai
            double temp_dist = heap->distancias[i];
            FaceRegistro *temp_face = heap->faces[i];
            
            heap->distancias[i] = heap->distancias[(i-1)/2];
            heap->faces[i] = heap->faces[(i-1)/2];
            
            heap->distancias[(i-1)/2] = temp_dist;
            heap->faces[(i-1)/2] = temp_face;
            
            i = (i-1)/2;
        }
    } else if (dist < heap->distancias[0]) {
        // Substitui o vizinho mais distante se o novo for mais próximo
        heap->distancias[0] = dist;
        heap->faces[0] = face;
        
        // Reorganiza o heap
        int i = 0;
        while (1) {
            int maior = i;
            int esq = 2*i + 1;
            int dir = 2*i + 2;
            
            if (esq < heap->tamanho && heap->distancias[esq] > heap->distancias[maior])
                maior = esq;
            if (dir < heap->tamanho && heap->distancias[dir] > heap->distancias[maior])
                maior = dir;
                
            if (maior == i) break;
            
            // Troca com o maior filho
            double temp_dist = heap->distancias[i];
            FaceRegistro *temp_face = heap->faces[i];
            
            heap->distancias[i] = heap->distancias[maior];
            heap->faces[i] = heap->faces[maior];
            
            heap->distancias[maior] = temp_dist;
            heap->faces[maior] = temp_face;
            
            i = maior;
        }
    }
}

// Busca recursiva pelos N vizinhos mais próximos
void _buscar_vizinhos(KDTree *arv, TreeNode *no, FaceRegistro *consulta, int profundidade, 
                     HeapVizinhos *heap) {
    if (no == NULL) return;
    
    // Calcula distância até o nó atual
    double dist = arv->distancia(no->key, consulta);
    printf("%s dist %.3f menor_dist %.3f comp %d\n", 
           ((FaceRegistro*)no->key)->id, dist, 
           (heap->tamanho > 0) ? heap->distancias[0] : dist,
           arv->comparador(consulta, no->key, profundidade % arv->dimensoes));
    
    // Adiciona ao heap se for um dos N mais próximos
    heap_adicionar_vizinho(heap, dist, (FaceRegistro*)no->key);
    
    int dim = profundidade % arv->dimensoes;
    int comp = arv->comparador(consulta, no->key, dim);
    
    // Decide qual subárvore explorar primeiro
    TreeNode *primeiro = comp < 0 ? no->esq : no->dir;
    TreeNode *segundo = comp < 0 ? no->dir : no->esq;
    
    _buscar_vizinhos(arv, primeiro, consulta, profundidade + 1, heap);
    
    // Verifica se precisa explorar a outra subárvore
    float diff = consulta->embedding[dim] - ((FaceRegistro*)no->key)->embedding[dim];
    if (heap->tamanho < heap->capacidade || diff * diff < heap->distancias[0]) {
        printf("tentando do outro lado %d\n", (int)(diff * diff));
        _buscar_vizinhos(arv, segundo, consulta, profundidade + 1, heap);
    }
}

// Busca os N vizinhos mais próximos
void buscar_n_vizinhos(KDTree *arv, FaceRegistro *consulta, int n, FaceRegistro **resultados) {
    HeapVizinhos heap;
    heap_inicializar(&heap, n);
    
    _buscar_vizinhos(arv, arv->raiz, consulta, 0, &heap);
    
    // Copia resultados para o array de saída
    for (int i = 0; i < heap.tamanho; i++) {
        resultados[i] = heap.faces[i];
    }
    
    free(heap.distancias);
    free(heap.faces);
}

// ==================== TESTES ====================

void testar_construcao() {
    KDTree arv;
    kdtree_inicializar(&arv, 2);
    
    float emb1[128] = {2.0, 3.0};
    float emb2[128] = {1.0, 1.0};
    
    FaceRegistro *reg1 = criar_registro(emb1, "Dourados");
    FaceRegistro *reg2 = criar_registro(emb2, "Campo Grande");
    
    assert(arv.raiz == NULL);
    assert(arv.dimensoes == 2);
    assert(arv.comparador(reg1, reg2, 0) == 1);
    assert(arv.comparador(reg1, reg2, 1) == 1);
    assert(strcmp(reg1->id, "Dourados") == 0);
    assert(strcmp(reg2->id, "Campo Grande") == 0);
    
    free(reg1);
    free(reg2);
}

void testar_busca() {
    KDTree arv;
    kdtree_inicializar(&arv, 2);
    
    float emb[128] = {0};
    
    emb[0] = 10; emb[1] = 10;
    kdtree_inserir(&arv, criar_registro(emb, "a"));
    
    emb[0] = 20; emb[1] = 20;
    kdtree_inserir(&arv, criar_registro(emb, "b"));
    
    emb[0] = 1; emb[1] = 10;
    kdtree_inserir(&arv, criar_registro(emb, "c"));
    
    emb[0] = 3; emb[1] = 5;
    kdtree_inserir(&arv, criar_registro(emb, "d"));
    
    emb[0] = 7; emb[1] = 15;
    kdtree_inserir(&arv, criar_registro(emb, "e"));
    
    emb[0] = 4; emb[1] = 11;
    kdtree_inserir(&arv, criar_registro(emb, "f"));
    
    // Testes de busca
    float emb_busca[128] = {7, 14};
    FaceRegistro *consulta = criar_registro(emb_busca, "x");
    FaceRegistro *resultados[1];
    
    buscar_n_vizinhos(&arv, consulta, 1, resultados);
    assert(strcmp(resultados[0]->id, "e") == 0);
    
    // Limpeza
    free(consulta);
    // (Implementar função de destruição da árvore para liberar memória)
}

int main() {
    testar_construcao();
    testar_busca();
    printf("SUCCESS!!\n");
    return 0;
}
