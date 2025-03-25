import torch
import torch.nn as nn
import torch.optim as optim
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

def generate_synthetic_dataset(num_samples=100, num_classes=10, image_size=(3, 224, 224)):
    """
    Gera um conjunto de dados sintético para demonstração
    """
    X = torch.randn(num_samples, *image_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

def train_model(model, X, y, epochs=10, learning_rate=0.001):
    """
    Treina o modelo com dados sintéticos
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Iniciando treinamento...")
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Imprimir progresso
        if (epoch + 1) % 2 == 0:
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).float().mean()
            print(f'Época [{epoch+1}/{epochs}], Perda: {loss.item():.4f}, Acurácia: {accuracy.item():.4f}')

    return model

def main():
    # Configurar semente para reprodutibilidade
    torch.manual_seed(42)
    np.random.seed(42)

    # Criar modelo
    model = EfficientNetBase(num_classes=10)

    # Gerar dados sintéticos
    X, y = generate_synthetic_dataset()

    # Treinar modelo
    trained_model = train_model(model, X, y)

    # Testar modelo com nova entrada
    test_input = torch.randn(1, 3, 224, 224)
    output = trained_model(test_input)
    print(f"\nFormato da saída para nova entrada: {output.shape}")
    print(f"Predições: {torch.softmax(output, dim=1)}")

if __name__ == "__main__":
    main()
