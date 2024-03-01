from math import ceil, sqrt

import torch


class VGG16Like(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(256, 256, kernel_size=3)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = torch.nn.Conv2d(256, 512, kernel_size=3)
        self.conv9 = torch.nn.Conv2d(512, 512, kernel_size=3)
        self.conv10 = torch.nn.Conv2d(512, 512, kernel_size=3)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = torch.nn.Conv2d(512, 512, kernel_size=3)
        self.conv12 = torch.nn.Conv2d(512, 512, kernel_size=3)
        self.conv13 = torch.nn.Conv2d(512, 512, kernel_size=3)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(512, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 1)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = images
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool1(x)

        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.pool2(x)

        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = torch.nn.functional.relu(self.conv7(x))
        x = self.pool3(x)

        x = torch.nn.functional.relu(self.conv8(x))
        x = torch.nn.functional.relu(self.conv9(x))
        x = torch.nn.functional.relu(self.conv10(x))
        x = self.pool4(x)

        x = torch.nn.functional.relu(self.conv11(x))
        x = torch.nn.functional.relu(self.conv12(x))
        x = torch.nn.functional.relu(self.conv13(x))
        x = self.pool5(x)

        x = x.view(-1, 512)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SmallVGG16Like(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.conv9 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.conv10 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.conv12 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.conv13 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = images
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool1(x)

        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.pool2(x)

        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = torch.nn.functional.relu(self.conv7(x))
        x = self.pool3(x)

        x = torch.nn.functional.relu(self.conv8(x))
        x = torch.nn.functional.relu(self.conv9(x))
        x = torch.nn.functional.relu(self.conv10(x))
        x = self.pool4(x)

        x = torch.nn.functional.relu(self.conv11(x))
        x = torch.nn.functional.relu(self.conv12(x))
        x = torch.nn.functional.relu(self.conv13(x))
        x = self.pool5(x)

        x = x.view(-1, 128)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class MiniCnn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(128 * 5 * 5, 256)
        self.fc2 = torch.nn.Linear(256, 1)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = images
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.pool5(x)

        x = x.view(-1, 128 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class InvertedResidual(torch.nn.Module):
    def __init__(
        self, input_channel: int, output_channel: int, stride: int, expand_ratio: int
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(input_channel * expand_ratio)
        self.use_res_connect = self.stride == 1 and input_channel == output_channel

        if expand_ratio == 1:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU6(inplace=True),
                torch.nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(output_channel),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU6(inplace=True),
                torch.nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU6(inplace=True),
                torch.nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(output_channel),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(torch.nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 216
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            [4, 32, 2, 2],
            [4, 64, 2, 2],
            [4, 96, 2, 2],
            # [4, 128, 2, 2],
            # [4, 128, 1, 1],
        ]

        # building first layer
        self.last_channel = last_channel
        self.features = [
            torch.nn.Sequential(
                torch.nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(input_channel),
                torch.nn.ReLU6(inplace=True),
            )
        ]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(
                            input_channel, output_channel, s, expand_ratio=t
                        )  # type: ignore
                    )
                else:
                    self.features.append(
                        InvertedResidual(
                            input_channel, output_channel, 1, expand_ratio=t
                        )  # type: ignore
                    )
                input_channel = output_channel

        # building last several layers
        self.features.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(self.last_channel),
                torch.nn.ReLU6(inplace=True),
            )
        )
        # make it nn.Sequential
        self.features_ = torch.nn.Sequential(*self.features)

        # building classifier
        self.classifier = torch.nn.Linear(self.last_channel, 1)

        self._initialize_weights()

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = images
        x = self.features_(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2Tab(torch.nn.Module):
    def __init__(self):
        super(MobileNetV2Tab, self).__init__()
        input_channel = 32
        last_channel = 216
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            [4, 32, 2, 2],
            [4, 64, 2, 2],
            [4, 96, 2, 2],
            # [4, 128, 2, 2],
            # [4, 128, 1, 1],
        ]

        # building first layer
        self.last_channel = last_channel
        self.features = [
            torch.nn.Sequential(
                torch.nn.Conv2d(6, input_channel, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(input_channel),
                torch.nn.ReLU6(inplace=True),
            )
        ]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(
                            input_channel, output_channel, s, expand_ratio=t
                        )  # type: ignore
                    )
                else:
                    self.features.append(
                        InvertedResidual(
                            input_channel, output_channel, 1, expand_ratio=t
                        )  # type: ignore
                    )
                input_channel = output_channel

        # building last several layers
        self.features.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(self.last_channel),
                torch.nn.ReLU6(inplace=True),
            )
        )
        # make it nn.Sequential
        self.features_ = torch.nn.Sequential(*self.features)

        # building classifier
        self.classifier = torch.nn.Linear(self.last_channel, 1)

        self._initialize_weights()

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            (
                images,
                tabular.unsqueeze(2)
                .unsqueeze(3)
                .expand(tabular.size(0), tabular.size(1), 224, 224),
            ),
            dim=1,
        )
        x = self.features_(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


MODEL_DICT = {
    "MiniCnn": MiniCnn,
    "MobileNetV2": MobileNetV2,
    "MobileNetV2Tab": MobileNetV2Tab,
    "SmallVGG16Like": SmallVGG16Like,
    "VGG16Like": VGG16Like,
}


if __name__ == "__main__":
    import torchinfo

    model = MobileNetV2Tab()
    torchinfo.summary(model)

    input_tensor = torch.rand(64, 3, 224, 224)
    tabular = torch.rand(64, 3)
    output = model(input_tensor, tabular)
    print(output.shape)
    print(output)
