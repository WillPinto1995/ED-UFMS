# ED-UFMS
Estruturas de Dados UFMS 
Trabalho 1 - Reconhecimento Facial com KD-Tree
Funcionalidades implementadas

Como usar:
1. Compilar o código em C: gcc -fPIC -shared -o libkdtree.so kdtree.c -lm
2. Execute: pip install fastapi uvicorn numpy pydantic



Funcionalidades implementadas:

KD-Tree com suporte para:

inserção de registros: embedding facial (128 floats) + identificador (100 caracteres).

busca KNN: busca eficiente dos K vizinhos mais próximos utilizando MinHeap.
