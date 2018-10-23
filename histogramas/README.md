## Laboratório de Introdução à Programação Paralela. 
### Introdução à Programação em CUDA.
Leitura sugerida: https://engineering.purdue.edu/~smidkiff/ece563/NVidiaGPUTeachingToolkit/Mod1/3rd-Edition-Chapter01-introduction.pdf
Slides: http://oxent2.ic.unicamp.br/sites/oxent2.ic.unicamp.br/files/7.Histogram.pdf
Vídeo-aulas: https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/876903830923

## Histogramas
Histogramas permitem representar graficamente em colunas um conjunto de dados dividido em clases, onde cada coluna representa uma clase e a altura da coluna representa a quantidade ou frequência com que a classe ocorre no conjunto de dados (veja mais em https://pt.wikipedia.org/wiki/Histograma).  

Algoritmo (coluna = Bin):
```
for (i=0; i < numColunas; i++)
   coluna[i] = 0;
for (i=0; i < data_size ; i++)
   coluna[calculaColuna(data[i])]++;
```

Algoritmo paralelo:
* Como distribuir os dados entre as threads?
* Como evitar condições de corrida?

Uma opção: use atomics
```C
__global__ void mykernel(int *addr) {
  atomicAdd_system(addr, 10);       // only available on devices with compute capability 6.x
}
```
