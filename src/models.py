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


class MobileNetv2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.9.0", "mobilenet_v2", pretrained=True
        )
        self.model.classifier[1] = torch.nn.Linear(1280, 1)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = images
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = MiniCnn()
    input_tensor = torch.rand(64, 3, 224, 224)
    output = model(input_tensor, None)
    print(output.shape)
    print(output)
