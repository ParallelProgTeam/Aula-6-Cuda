# Aula-6-Cuda
# Laboratório de Introdução à Programação Paralela. Introdução à Programação em CUDA.

**CUDA** é uma plataforma de computação paralela e modelo de programação da NVIDIA para a programação de propósito geral usando GPUs da marca. Programas CUDA são heterogêneos: parte do programa é executado na CPU, enquanto trechos computacionalmente caros executam na GPU. 
As funções que são lançadas na CPU (chamado de *host* na terminologia CUDA) para execução na GPU (ou *device*) são chamadas de *kernels* e se declaram como **\_\_global\_\_**.

## Primeiro programa: Hello World! (serial)

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

##Primeiro programa: Hello World! (paralelo!)

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

Qual é a saída da execução? 1 vez o texto "Hello World!", certo?

Agora troque a linha ```cpp	mykernel<<<1,1>>>(); ``` por ```cpp	mykernel<<<1,8>>>();```
Qual é a saída da execução dessa vez? 
Exatamente 8 vezes o texto "Hello World!", uma vez por cada thread executada na GPU.

cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l

e no campo core id, quantas threads lógicas podem ser executadas.

# cat /proc/cpuinfo | egrep "core id|physical id" | tr -d "\n" | sed s/physical/\\nphysical/g | grep -v ^$ | sort | uniq | wc -l

Um aparte, o que são threads lógicas? Veja no documento em anexo o texto sobre SMT.

#pragma omp parallel é uma diretiva de compilação do OpenMP que indica que o bloco de código será executado em paralelo.
Paralelizando um laço

Como dito anteriormente, o foco do OpenMP é a paralelização de laços. Normalmente em um algoritmo, as estruturas de loop representam a porção de código com maior custo computacional.

Sendo assim, existe a diretiva de compilação #pragma omp for em OpenMP. A seguir, são apresentadas duas formas de paralelizar um laço:

#pragma omp parallel
{
    #pragma omp for
    for(int i = 0; i < n; i++)
    {
        cout << i << endl;
    }
}

Forma reduzida:

#pragma omp parallel for
for(int i = 0; i < n; i++)
{
    cout << i << endl;
}

É possível perceber que a execução do laço não segue a ordenação do vetor percorrido.
Usando funções da API do OpenMP

O OpenMP fornece diversas funções que permitem definir a quantidade de threads, descobrir o identificador (thread ID) da thread em execução, criar e utilizar mecanismos de sincronização, e várias outras funcionalides.

Para usar as funções da API é preciso incluir o header file omp.h no programa. No exemplo a seguir, são utilizadas duas funções da API:

#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    omp_set_num_threads(2);

    #pragma omp parallel for
    for(int i = 0; i < 10; i++)
    {
        printf("Thread ID: %d -- index: %d\n",
            omp_get_thread_num(), i);
    }
}

A função omp_set_num_threads define o número de threads que serão utilizadas durante a execução em paralelo. Já a função omp_get_thread_num retorna o identificador único da thread que está em execução no momento.
Problema 1 - SAXPY

Para ver a descrição do problema clique aqui
Problema 2 - Processar uma imagem

Para ver a descrição do problema clique aqui
Sincronização com OpenMP

Um problema muito comum em algoritmos paralelos é o acesso concorrente à um mesmo recurso, o que gera uma condição de corrida. Um exemplo é a redução de um vetor de inteiros.

Código sequencial:

#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    int sum = 0;
    for(int i = 0; i < n; i++)
    {
        sum += array[i];
    }

    cout << sum << endl;
}
