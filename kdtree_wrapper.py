import ctypes
from ctypes import Structure, POINTER, c_float, c_char, c_int, c_void_p

class FaceReg(Structure):
    _fields_ = [
        ("embedding", c_float * 128),
        ("id", c_char * 100)
    ]

class TreeNode(Structure):
    pass

TreeNode._fields_ = [
    ("key", c_void_p),
    ("esq", POINTER(TreeNode)),
    ("dir", POINTER(TreeNode))
]

class KDTree(Structure):
    _fields_ = [
        ("k", c_int),
        ("dist", c_void_p),
        ("cmp", c_void_p),
        ("raiz", POINTER(TreeNode))
    ]

class KDTreeWrapper:
    def __init__(self):
        self.lib = ctypes.CDLL("/mnt/v/facom/ed/trabalho/tabalho_ed/api/libkdtree.so")
        self._setup_functions()

    def _setup_functions(self):
        # Configuração dos tipos de argumentos e retorno
        self.lib.kdtree_construir.argtypes = []
        self.lib.kdtree_construir.restype = None
        
        self.lib.inserir_ponto.argtypes = [FaceReg]
        self.lib.inserir_ponto.restype = None
        
        self.lib.buscar_n_vizinhos.argtypes = [
            POINTER(KDTree),
            POINTER(FaceReg),
            c_int,
            POINTER(POINTER(FaceReg))
        ]
        self.lib.buscar_n_vizinhos.restype = None
        
        self.lib.get_tree.restype = POINTER(KDTree)

    def initialize(self):
        """Inicializa a KDTree"""
        self.lib.kdtree_construir()

    def build_tree(self):
        """Constrói a estrutura da árvore"""
        self.lib.kdtree_construir()

    def insert_face(self, embedding: list, face_id: str):
        """Insere uma face na árvore"""
        if len(embedding) != 128:
            raise ValueError("Embedding deve ter 128 dimensões")
        
        emb_array = (c_float * 128)(*embedding)
        face_bytes = face_id.encode('utf-8')[:99]
        face_reg = FaceReg(embedding=emb_array, id=face_bytes)
        self.lib.inserir_ponto(face_reg)

    def search_faces(self, embedding: list, n: int = 5):
        """Busca as n faces mais similares"""
        emb_array = (c_float * 128)(*embedding)
        query = FaceReg(embedding=emb_array, id=b"")
        
        tree_ptr = self.lib.get_tree()
        result_type = POINTER(FaceReg) * n
        results = result_type()
        
        self.lib.buscar_n_vizinhos(tree_ptr, query, n, results)
        
        return [
            (results[i].contents.id.decode('utf-8').rstrip('\x00'), 
             1.0 - getattr(results[i].contents, 'distancia', 0))
            for i in range(n) if results[i]
        ]
