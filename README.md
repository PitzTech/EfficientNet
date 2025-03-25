# Instrução de Instalação do Projeto EfficientNet

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
