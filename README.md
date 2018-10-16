## Laboratório de Introdução à Programação Paralela. 
### Introdução à Programação em CUDA.

**CUDA** é uma plataforma de computação paralela e modelo de programação da NVIDIA para a programação de propósito geral usando GPUs da marca. Programas CUDA são heterogêneos: parte do programa é executado na CPU, enquanto trechos computacionalmente caros executam na GPU. 

### Primeiro programa: Hello World! (serial)

Vamos testar o *nvcc* (compilador NVIDIA cc) de sua máquina para garantir que está tudo certo. Crie o arquivo *main.cu* contendo o seguinte código:
```cpp
#include <iostream>
using namespace std;
int main() {
		printf("Hello World!\n");
		return 0;
	}
```
Para compilar e executar o programa:
```bash
$ nvcc main.cu -o output
$ ./output
```

###Primeiro programa: Hello World! (paralelo!)
As funções que são lançadas na CPU (chamado de *host* na terminologia CUDA) para execução na GPU (ou *device*) são chamadas de *kernels* e se declaram como **\_\_global\_\_**. O número de threads CUDA que executarão o kernel em uma determinada chamada é especificado usando 
a sintaxe de configuração da execução : <<<...>>>

Crie o arquivo main.cu contendo o seguinte código e compile-o:
```cpp
__global__ void mykernel(void) {
	}

	int main(void) {
		mykernel<<<1,1>>>();
		printf("Hello World!\n");
		return 0;
	}
```

Qual é a saída da execução? 

Agora troque a linha ```mykernel<<<1,1>>>(); ``` por ```mykernel<<<1,8>>>();```
Qual é a saída da execução dessa vez? 
Exatamente 8 vezes o texto "Hello World!", uma vez por cada thread executada na GPU.

A programação CUDA segue um modelo de execução chamado de SIMT (Single Instruction, Multiple Thread). No lugar do modelo SIMD descrito na taxonomia de Flynn em que uma thread de controle processaria vários elementos de dado (como nos processadores vetoriais), em SIMT grupos de threads executam a mesma instrução simultâneamente sobre diferentes dados. Em CUDA, esses grupos são chamados de *warps* e até agora são formados estritamente por 32 threads. 

Threads podem ser distinguidas por meio de sua id de thread, que pode ser acessada através da variável threadIdx. 
O runtime CUDA fornece diversas funções e que permitem definir a quantidade de threads, criar e utilizar mecanismos de sincronização, e várias outras funcionalidades.

## Problema 1 - SAXPY

[Para ver a descrição do problema clique aqui](./saxpy)

Controle de divergências.
Já que grupos de 32 threads (um warp) executam a mesma instrução, como são tratadas as divergências em CUDA?
Isto pode danificar o desempenho, a programação, no possível, deve levar em conta o nível de warps para maximizar o desempenho. 
[https://devblogs.nvidia.com/using-cuda-warp-level-primitives/]


Mais informação: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
