#!/bin/bash

# Verifica se o Python está instalado
if ! command -v python3 &> /dev/null
then
    echo "Python 3 não encontrado. Por favor, instale o Python."
    exit 1
fi

# Criação de ambiente virtual
python3 -m venv efficientnet_env

# Ativação do ambiente virtual
source efficientnet_env/bin/activate

# Atualização do pip
pip install --upgrade pip

# Instalação das dependências
pip install -r requirements.txt

# Verificação da instalação do PyTorch
python3 -c "import torch; print('Torch versão:', torch.__version__)"

echo "Instalação concluída com sucesso!"
