from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import ctypes
from ctypes import c_float, c_char
from kdtree_wrapper import KDTreeWrapper

app = FastAPI(title="Facial Recognition API")

# Inicializa o wrapper da KDTree
kdtree = KDTreeWrapper()

# Modelos Pydantic para validação
class FaceInput(BaseModel):
    embedding: List[float]
    id: str

class FaceSearchResult(BaseModel):
    id: str
    similarity: float

@app.on_event("startup")
def startup_event():
    """Inicializa a KDTree quando a API começa"""
    kdtree.initialize()

@app.post("/build-tree", summary="Constroi a árvore KD")
def build_tree():
    try:
        kdtree.build_tree()
        return {"message": "KDTree construída com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert-face", summary="Insere uma face na árvore")
def insert_face(face: FaceInput):
    try:
        if len(face.embedding) != 128:
            raise ValueError("Embedding deve ter 128 dimensões")
        
        kdtree.insert_face(face.embedding, face.id)
        return {"message": f"Face '{face.id}' inserida com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search-faces", summary="Busca faces similares", response_model=List[FaceSearchResult])
def search_faces(embedding: List[float], n: int = 5):
    try:
        if len(embedding) != 128:
            raise ValueError("Embedding deve ter 128 dimensões")
        
        results = kdtree.search_faces(embedding, n)
        return [{"id": r[0], "similarity": r[1]} for r in results]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
