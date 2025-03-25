# Instrução de Instalação do Projeto EfficientNet

# Instale PyTorch com suporte a CUDA
# Visite https://pytorch.org/get-started/locally/ para versão exata

## Pré-requisitos
- Python 3.8 ou superior
- pip
- venv (ambiente virtual)

## Passos de Instalação

1. Clone o repositório:
```bash
git clone https://github.com/PitzTech/EfficientNet.git
cd EfficientNet
```

2. Crie e ative o ambiente virtual:
```bash
python3 -m venv efficientnet_env
source efficientnet_env/bin/activate  # No Windows use: efficientnet_env\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Verificação de instalação:
```bash
python3 -c "import torch; print('Torch versão:', torch.__version__)"
```

## Desativação do Ambiente Virtual
```bash
deactivate
```

##Exemplo de saída

```bash
C:\Users\victo\Downloads\ia trabalho\code>py EfficientNet-Cifar10.py
100.0%
Iniciando treinamento...
Época [1/10], Perda de Treinamento: 1.7156, Acurácia de Treinamento: 36.77%, Acurácia de Teste: 47.49%
Época [2/10], Perda de Treinamento: 1.3686, Acurácia de Treinamento: 50.61%, Acurácia de Teste: 57.48%
Época [3/10], Perda de Treinamento: 1.2161, Acurácia de Treinamento: 56.71%, Acurácia de Teste: 61.95%
Época [4/10], Perda de Treinamento: 1.1116, Acurácia de Treinamento: 60.38%, Acurácia de Teste: 65.34%
Época [5/10], Perda de Treinamento: 1.0279, Acurácia de Treinamento: 63.59%, Acurácia de Teste: 66.90%
Época [6/10], Perda de Treinamento: 0.9706, Acurácia de Treinamento: 66.07%, Acurácia de Teste: 69.39%
Época [7/10], Perda de Treinamento: 0.9230, Acurácia de Treinamento: 67.58%, Acurácia de Teste: 71.21%
Época [8/10], Perda de Treinamento: 0.8885, Acurácia de Treinamento: 68.84%, Acurácia de Teste: 71.59%
Época [9/10], Perda de Treinamento: 0.8494, Acurácia de Treinamento: 70.25%, Acurácia de Teste: 73.04%
Época [10/10], Perda de Treinamento: 0.8234, Acurácia de Treinamento: 71.29%, Acurácia de Teste: 73.50%

Imagem de Teste:
Classe Verdadeira: cat
Classe Predita: cat
Probabilidades: tensor([[0.0044, 0.0018, 0.0100, 0.5243, 0.0031, 0.4040, 0.0097, 0.0090, 0.0306, 0.0032]])
```
