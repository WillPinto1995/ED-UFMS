from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from kdtree_wrapper import lib, TReg
from ctypes import c_float, c_char

app = FastAPI()

# =====================================
# Modelo de entrada para API
# =====================================

class PessoaEntrada(BaseModel):
    embedding: List[float]  # Vetor de 128 floats
    nome: str               # Nome ou identificador

class ConsultaEntrada(BaseModel):
    embedding: List[float]  # Vetor de 128 floats
    k: int = 2              # Número de vizinhos

# =====================================
# Rotas da API
# =====================================

@app.post("/construir-arvore")
def construir_arvore():
    lib.kdtree_construir()
    return {"mensagem": "Árvore KD construída com sucesso."}

@app.post("/inserir")
def inserir_pessoa(pessoa: PessoaEntrada):
    emb_array = (c_float * 128)(*pessoa.embedding)
    nome_bytes = pessoa.nome.encode('utf-8')[:99]
    novo_ponto = TReg(embedding=emb_array, nome=nome_bytes)
    lib.inserir_ponto(novo_ponto)
    return {"mensagem": f"Pessoa '{pessoa.nome}' inserida com sucesso."}

@app.post("/buscar")
def buscar_vizinhos(consulta: ConsultaEntrada):
    emb_array = (c_float * 128)(*consulta.embedding)
    nome_bytes = b"consulta"
    consulta_ponto = TReg(embedding=emb_array, nome=nome_bytes)
    
    # Chamada para C: supondo função lib.buscar_n_proximos(arvore, TReg, k)
    arvore = lib.get_tree()
    resultados = lib.buscar_n_proximos(arvore, consulta_ponto, consulta.k)

    vizinhos = []
    for i in range(consulta.k):
        vizinho = resultados[i]
        nome = bytes(vizinho.nome).decode('utf-8').rstrip('\x00')
        vizinhos.append({
            "nome": nome
            # Opcional: incluir embedding, mas normalmente não é necessário expor.
        })

    return {
        "consulta": consulta.nome if hasattr(consulta, "nome") else "Consulta",
        "vizinhos": vizinhos
    }

@app.get("/")
def home():
    return {"mensagem": "API de Reconhecimento Facial com KD-Tree ativa!"}
