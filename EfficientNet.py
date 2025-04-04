import torch
import torch.nn as nn
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
    def __init__(self, num_classes=1000, width_multiplier=1.0, depth_multiplier=1.0):
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

# Exemplo de uso
def main():
    # Criar modelo
    model = EfficientNetBase(num_classes=10)

    # Exemplo de entrada
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Formato da saída: {output.shape}")

if __name__ == "__main__":
    main()
