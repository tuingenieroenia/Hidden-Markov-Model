# Hidden Markov Model

Este proyecto implementa un modelo oculto de Markov (HMM) con algoritmos como Forward-Backward y Viterbi, utilizados comúnmente para la secuenciación y predicción en series temporales. El código está diseñado para manejar cálculos probabilísticos y decodificación en un HMM dado un conjunto de estados, observaciones y probabilidades de transición.

## Estructura del Proyecto

El repositorio contiene los siguientes archivos principales:

- `example_hmm.py`: Un ejemplo básico que muestra cómo implementar un modelo oculto de Markov utilizando los métodos y algoritmos de este repositorio.
- `f_v_2.py`: Contiene una implementación alternativa del algoritmo Forward-Backward para decodificar secuencias observadas.
- `foward_viterbi.py`: Implementación del algoritmo de Viterbi para encontrar la secuencia más probable de estados ocultos.
- `methods.py`: Contiene varios métodos para trabajar con modelos ocultos de Markov, incluyendo cálculos de probabilidad y transición.

## Requisitos

Para ejecutar el código, necesitas tener instalado Python 3.x. Puedes instalar las dependencias necesarias con:

```bash
pip install numpy
