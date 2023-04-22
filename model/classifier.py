from omegaconf import DictConfig
from torch import Tensor, nn


class SimpleClassifier(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        padding = (cfg.model.kernel_size - 1) // 2
        self.pre_conv = nn.Conv2d(
            in_channels=cfg.model.input_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.convs = nn.ModuleList(
            nn.Conv2d(
                in_channels=cfg.model.hidden_channels,
                out_channels=cfg.model.hidden_channels,
                kernel_size=cfg.model.kernel_size,
                padding=padding,
                padding_mode="reflect",
            )
            for _ in range(8)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorms = nn.ModuleList(
            nn.BatchNorm2d(num_features=cfg.model.hidden_channels) for _ in range(8)
        )
        self.linear = nn.Linear(
            in_features=cfg.model.hidden_channels * 49,
            out_features=cfg.model.num_classes,
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, data: Tensor) -> Tensor:
        hidden = self.pre_conv(data)
        hidden = self.relu(hidden)
        data = data + hidden

        for i, (conv, batchnorm) in enumerate(zip(self.convs, self.batchnorms)):
            hidden = conv(data)
            hidden = batchnorm(hidden)
            hidden = self.relu(hidden)
            data = data + hidden
            if i % 4 == 3:
                data = self.max_pool(data)

        data = self.flatten(data)
        data = self.linear(data)
        return data
