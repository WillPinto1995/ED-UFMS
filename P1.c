#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =======================================
   Estrutura que representa um indivíduo
   ======================================= */
typedef struct {
    float features[128];   // Vetor de características (embedding facial)
    char nome[100];        // Identificação
} Pessoa;

/* =======================================
   Estrutura do Nó da Árvore KD
   ======================================= */
typedef struct KDNo {
    Pessoa* pessoa;
    struct KDNo* esquerda;
    struct KDNo* direita;
} KDNo;

/* =======================================
   Estrutura da Árvore KD
   ======================================= */
typedef struct {
    KDNo* raiz;
    int dimensao;  // Sempre 128 neste caso
} KDArvore;

/* =======================================
   Estruturas da MinHeap
   ======================================= */
typedef struct {
    double distancia;
    Pessoa* pessoa;
} ElementoHeap;

typedef struct {
    ElementoHeap* elementos;
    int capacidade;
    int tamanho;
} MinHeap;

/* =======================================
   Funções Auxiliares Gerais
   ======================================= */

// Calcula a distância euclidiana ao quadrado
double calcular_distancia(Pessoa* p1, Pessoa* p2) {
    double soma = 0.0;
    for (int i = 0; i < 128; i++) {
        double diff = p1->features[i] - p2->features[i];
        soma += diff * diff;
    }
    return soma;
}

// Compara duas pessoas na dimensão pos
int comparar_dimensao(Pessoa* p1, Pessoa* p2, int pos) {
    if (p1->features[pos] < p2->features[pos]) return -1;
    if (p1->features[pos] > p2->features[pos]) return 1;
    return 0;
}

// Cria uma nova pessoa
Pessoa* criar_pessoa(float features[], const char* nome) {
    Pessoa* nova = malloc(sizeof(Pessoa));
    memcpy(nova->features, features, sizeof(float) * 128);
    strncpy(nova->nome, nome, 99);
    nova->nome[99] = '\0';
    return nova;
}

/* =======================================
   Funções da MinHeap
   ======================================= */

// Cria uma MinHeap com capacidade dada
MinHeap* inicializar_heap(int capacidade) {
    MinHeap* heap = malloc(sizeof(MinHeap));
    heap->elementos = malloc(sizeof(ElementoHeap) * capacidade);
    heap->capacidade = capacidade;
    heap->tamanho = 0;
    return heap;
}

// Libera a MinHeap
void destruir_heap(MinHeap* heap) {
    free(heap->elementos);
    free(heap);
}

// Troca dois elementos no heap
void trocar_elementos(ElementoHeap* a, ElementoHeap* b) {
    ElementoHeap temp = *a;
    *a = *b;
    *b = temp;
}

// Ajusta heap para manter propriedade de max-heap
void ajustar_heap_acima(MinHeap* heap, int indice) {
    while (indice > 0) {
        int pai = (indice - 1) / 2;
        if (heap->elementos[indice].distancia > heap->elementos[pai].distancia) {
            trocar_elementos(&heap->elementos[indice], &heap->elementos[pai]);
            indice = pai;
        } else {
            break;
        }
    }
}

void ajustar_heap_abaixo(MinHeap* heap, int indice) {
    while (1) {
        int maior = indice;
        int esq = 2 * indice + 1;
        int dir = 2 * indice + 2;

        if (esq < heap->tamanho && heap->elementos[esq].distancia > heap->elementos[maior].distancia)
            maior = esq;
        if (dir < heap->tamanho && heap->elementos[dir].distancia > heap->elementos[maior].distancia)
            maior = dir;

        if (maior != indice) {
            trocar_elementos(&heap->elementos[indice], &heap->elementos[maior]);
            indice = maior;
        } else {
            break;
        }
    }
}

// Insere na MinHeap
void inserir_na_heap(MinHeap* heap, double distancia, Pessoa* pessoa) {
    if (heap->tamanho < heap->capacidade) {
        heap->elementos[heap->tamanho].distancia = distancia;
        heap->elementos[heap->tamanho].pessoa = pessoa;
        ajustar_heap_acima(heap, heap->tamanho);
        heap->tamanho++;
    } else if (distancia < heap->elementos[0].distancia) {
        heap->elementos[0].distancia = distancia;
        heap->elementos[0].pessoa = pessoa;
        ajustar_heap_abaixo(heap, 0);
    }
}

/* =======================================
   Funções da KD-Tree
   ======================================= */

// Insere recursivamente na árvore
void inserir_kd_recursivo(KDNo** raiz, Pessoa* pessoa, int profundidade, int dimensao) {
    if (*raiz == NULL) {
        *raiz = malloc(sizeof(KDNo));
        (*raiz)->pessoa = pessoa;
        (*raiz)->esquerda = NULL;
        (*raiz)->direita = NULL;
    } else {
        int pos = profundidade % dimensao;
        if (comparar_dimensao(pessoa, (*raiz)->pessoa, pos) < 0) {
            inserir_kd_recursivo(&(*raiz)->esquerda, pessoa, profundidade + 1, dimensao);
        } else {
            inserir_kd_recursivo(&(*raiz)->direita, pessoa, profundidade + 1, dimensao);
        }
    }
}

// Insere na árvore
void inserir_na_kdtree(KDArvore* arvore, Pessoa* pessoa) {
    inserir_kd_recursivo(&arvore->raiz, pessoa, 0, arvore->dimensao);
}

// Busca os K vizinhos mais próximos
void buscar_knn_recursivo(KDNo* no, Pessoa* consulta, int profundidade, int dimensao, MinHeap* heap) {
    if (no == NULL) return;

    double dist = calcular_distancia(consulta, no->pessoa);
    inserir_na_heap(heap, dist, no->pessoa);

    int pos = profundidade % dimensao;
    int comp = comparar_dimensao(consulta, no->pessoa, pos);

    KDNo* principal = comp < 0 ? no->esquerda : no->direita;
    KDNo* oposto = comp < 0 ? no->direita : no->esquerda;

    buscar_knn_recursivo(principal, consulta, profundidade + 1, dimensao, heap);

    double diff = consulta->features[pos] - no->pessoa->features[pos];
    if (heap->tamanho < heap->capacidade || (diff * diff) < heap->elementos[0].distancia) {
        buscar_knn_recursivo(oposto, consulta, profundidade + 1, dimensao, heap);
    }
}

// Busca os K vizinhos mais próximos da consulta
void buscar_k_vizinhos(KDArvore* arvore, Pessoa* consulta, Pessoa** vizinhos, int k) {
    MinHeap* heap = inicializar_heap(k);
    buscar_knn_recursivo(arvore->raiz, consulta, 0, arvore->dimensao, heap);

    for (int i = 0; i < heap->tamanho; i++) {
        vizinhos[i] = heap->elementos[i].pessoa;
    }

    destruir_heap(heap);
}

/* =======================================
   Função Principal de Teste
   ======================================= */

int main() {
    KDArvore arvore;
    arvore.raiz = NULL;
    arvore.dimensao = 128;

    // Criar vetores fictícios
    float v1[128], v2[128], v3[128], vq[128];
    for (int i = 0; i < 128; i++) {
        v1[i] = 0.1f + i * 0.001f;
        v2[i] = 0.2f + i * 0.001f;
        v3[i] = 0.5f + i * 0.002f;
        vq[i] = 0.15f + i * 0.001f;
    }

    Pessoa* p1 = criar_pessoa(v1, "Individuo_A");
    Pessoa* p2 = criar_pessoa(v2, "Individuo_B");
    Pessoa* p3 = criar_pessoa(v3, "Individuo_C");

    inserir_na_kdtree(&arvore, p1);
    inserir_na_kdtree(&arvore, p2);
    inserir_na_kdtree(&arvore, p3);

    printf("3 registros foram inseridos na árvore KD.\n");

    Pessoa consulta;
    memcpy(consulta.features, vq, sizeof(vq));
    strcpy(consulta.nome, "Consulta");

    int k = 2;
    Pessoa* vizinhos[k];

    buscar_k_vizinhos(&arvore, &consulta, vizinhos, k);

    printf("Os %d vizinhos mais próximos de %s são:\n", k, consulta.nome);
    for (int i = 0; i < k; i++) {
        double dist = sqrt(calcular_distancia(&consulta, vizinhos[i]));
        printf("%d: %s (Distância: %.4f)\n", i + 1, vizinhos[i]->nome, dist);
    }

    return 0;
}
