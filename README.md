# ED-UFMS
Estruturas de Dados UFMS 
Trabalho 1 - Reconhecimento Facial com KD-Tree
Funcionalidades implementadas

Como usar:

gcc (para compilar a biblioteca C)

python >=3.8

pip

uvicorn, fastapi, pydantic, numpy



Inserção de registros contendo:

embedding facial: vetor de 128 floats.

identificador: string de até 100 caracteres.

Implementação da busca dos K vizinhos mais próximos utilizando:

Estratégia de poda típica da KD-Tree.

Utilização de MinHeap para manter os K menores vizinhos.
