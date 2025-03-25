import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, stride, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()

        self.skip_connection = stride == 1 and in_channels == out_channels

        # Expansão
        expanded_channels = in_channels * expansion_factor

        # Camadas
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        ) if expansion_factor != 1 else nn.Identity()

        # Convolução depthwise
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels,
                      kernel_size, stride,
                      padding=kernel_size//2,
                      groups=expanded_channels,
                      bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )

        # Convolução pontual
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        x = self.dropout(x)

        if self.skip_connection:
            x += residual

        return x

class EfficientNetBase(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=1.0, depth_multiplier=1.0):
        super(EfficientNetBase, self).__init__()

        # Configurações base inspiradas no EfficientNet-B0
        base_config = [
            # [in_channels, out_channels, expansion, kernel_size, stride]
            [32, 16, 1, 3, 1],
            [16, 24, 6, 3, 2],
            [24, 40, 6, 5, 2],
            [40, 80, 6, 3, 2],
            [80, 112, 6, 5, 1],
            [112, 192, 6, 5, 2],
            [192, 320, 6, 3, 1]
        ]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Blocos MBConv
        layers = []
        in_channels = 32
        for block_cfg in base_config:
            # Ajuste para width e depth multiplier
            out_channels = int(block_cfg[1] * width_multiplier)
            num_blocks = math.ceil(block_cfg[0] * depth_multiplier)

            layers.append(
                MBConvBlock(
                    in_channels,
                    out_channels,
                    block_cfg[2],
                    block_cfg[3],
                    block_cfg[4]
                )
            )

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Cabeçalho de classificação
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


def load_cifar10(batch_size=64):
    """
    Carrega o conjunto de dados CIFAR-10
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return trainloader, testloader

def train_model(model, trainloader, testloader, epochs=10, learning_rate=0.001):
    """
    Treina o modelo com CIFAR-10
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print("Iniciando treinamento...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Estatísticas de treinamento
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Avaliação no conjunto de teste
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        # Imprimir progresso
        train_accuracy = 100 * correct / total
        test_accuracy = 100 * test_correct / test_total
        print(f'Época [{epoch+1}/{epochs}], '
              f'Perda de Treinamento: {running_loss/len(trainloader):.4f}, '
              f'Acurácia de Treinamento: {train_accuracy:.2f}%, '
              f'Acurácia de Teste: {test_accuracy:.2f}%')

    return model

def main():
    # Configurar semente para reprodutibilidade
    torch.manual_seed(42)
    np.random.seed(42)

    # Criar modelo
    model = EfficientNetBase(num_classes=10)

    # Carregar dados
    trainloader, testloader = load_cifar10()

    # Treinar modelo
    trained_model = train_model(model, trainloader, testloader)

    # Testar modelo com uma imagem do conjunto de teste
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    test_image, true_label = testset[0]
    test_input = test_image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(test_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"\nImagem de Teste:")
    print(f"Classe Verdadeira: {class_names[true_label]}")
    print(f"Classe Predita: {class_names[predicted_class]}")
    print(f"Probabilidades: {probabilities}")

if __name__ == "__main__":
    main()
