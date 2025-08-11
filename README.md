# Classificação de Glaucoma – Reconhecimento de Imagens em Python

## Visão Geral

Este projeto implementa um sistema de classificação de imagens de retina para auxiliar no diagnóstico de glaucoma.
Ele utiliza redes neurais convolucionais (CNNs) pré-treinadas e adaptadas para a tarefa, explorando diferentes arquiteturas modernas de visão computacional.

## Estrutura do Projeto

* **`Treinamento_final/`** – Modelos de classificação otimizados, utilizando como base:

  * ResNet
  * DenseNet-121
  * GoogLeNet
* **`Treinamento_inicial/`** – Modelos experimentais com diferentes arquiteturas, incluindo:

  * AlexNet
  * DenseNet-121
  * EfficientNet
  * GoogLeNet
  * MobileNetV2
  * ResNeXt
  * SENet
  * SqueezeNet
  * VGGNet
  * Xception
* **`Versoes_Antigas/`** – Implementações incompletas ou descontinuadas.

## Funcionalidades

* Treinamento de modelos de classificação de imagens a partir de retinografias.
* Suporte a múltiplas arquiteturas de CNN.
* Configuração flexível dos caminhos (paths) para o banco de dados de imagens.
* Scripts separados para experimentos iniciais e treinamento final.

## Pré-requisitos

* Python 3.x
* Bibliotecas:

  * PyTorch
  * torchvision
  * NumPy
  * PIL/Pillow
  * scikit-learn
  * matplotlib

## Como Executar

1. Clone o repositório e acesse o diretório:

   ```bash
   git clone <url-do-repositorio>
   cd IA-Reconhecimento-de-Imagens-Python
   ```
2. Configure o caminho do banco de dados de imagens nos scripts desejados (presentes em `Treinamento_final/` ou `Treinamento_inicial/`).
3. Execute o script do modelo desejado:

   ```bash
   python Treinamento_final/Resnet-18.py
   ```
4. Acompanhe métricas de treino e avaliação exibidas no console ou salvas em arquivos.

## Habilidades demonstradas

* Implementação de **classificação de imagens médicas** com CNNs.
* Fine-tuning de modelos pré-treinados.
* Estruturação modular de código para experimentos e produção.
* Manipulação de datasets de imagens com PyTorch.
* Avaliação de modelos usando métricas de classificação.
