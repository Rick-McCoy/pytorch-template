from torch import Tensor, nn

from config.config import Config


class SimpleResidualBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        padding = (cfg.model.kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        self.batchnorm = nn.BatchNorm2d(num_features=cfg.model.hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, data: Tensor) -> Tensor:
        hidden = self.conv(data)
        hidden = self.batchnorm(hidden)
        hidden = self.relu(hidden)
        data = data + hidden
        return data


class SimpleClassifier(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        padding = (cfg.model.kernel_size - 1) // 2
        self.pre_conv = nn.Conv2d(
            in_channels=cfg.model.input_channels,
            out_channels=cfg.model.hidden_channels,
            kernel_size=cfg.model.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
        # max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # resblock_list: List[nn.Module] = [
        #     SimpleResidualBlock(cfg) for _ in range(cfg.model.num_layers)
        # ]
        # resblock_list.insert(cfg.model.num_layers // 2, max_pool)
        # resblock_list.append(max_pool)
        # self.resblocks = nn.Sequential(*resblock_list)
        self.linear = nn.Linear(
            in_features=(
                cfg.model.input_channels * (cfg.model.h // 4) * (cfg.model.w // 4) * 16
            ),
            out_features=cfg.model.num_classes,
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, data: Tensor) -> Tensor:
        # hidden = self.pre_conv(data)
        # hidden = self.relu(hidden)
        # data = data + hidden

        # data = self.resblocks(data)

        data = self.flatten(data)
        data = self.linear(data)
        return data
