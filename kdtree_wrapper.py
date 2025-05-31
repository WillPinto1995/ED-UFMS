import ctypes
from ctypes import Structure, POINTER, c_float, c_char, c_int, c_double

# =========================================
# Estruturas compatíveis com a biblioteca C
# =========================================

class Registro(Structure):
    _fields_ = [
        ("embedding", c_float * 128),
        ("nome", c_char * 100)
    ]

class NoKD(Structure):
    pass

NoKD._fields_ = [
    ("key", ctypes.c_void_p),
    ("esq", POINTER(NoKD)),
    ("dir", POINTER(NoKD))
]

class ArvoreKD(Structure):
    _fields_ = [
        ("k", c_int),
        ("dist", ctypes.CFUNCTYPE(c_double, ctypes.c_void_p, ctypes.c_void_p)),
        ("cmp", ctypes.CFUNCTYPE(c_int, ctypes.c_void_p, ctypes.c_void_p, c_int)),
        ("raiz", POINTER(NoKD))
    ]

# =========================================
# Carregando a biblioteca C
# =========================================

kdtree = ctypes.CDLL("./libkdtree.so")

# =========================================
# Definindo as funções da biblioteca C
# =========================================

# Construção da árvore
kdtree.kdtree_construir.argtypes = []
kdtree.kdtree_construir.restype = None

# Inserção de um registro
kdtree.inserir_ponto.argtypes = [Registro]
kdtree.inserir_ponto.restype = None

# Recupera ponteiro para árvore global
kdtree.get_tree.restype = POINTER(ArvoreKD)

# Busca de N vizinhos mais próximos
kdtree.buscar_n_vizinhos_proximos.argtypes = [
    POINTER(ArvoreKD), Registro, POINTER(Registro), c_int
]
kdtree.buscar_n_vizinhos_proximos.restype = None
