import torch.nn as nn


class SpectrogramCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super().__init__()
        # Convolutional layers
        # input is 64 x 32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce dimensions

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Input features = num_channels_last_conv * pool_output_height * pool_output_width
        self.fc1 = nn.Linear(int(64 * 64 / 2 * 32 / 2), num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [Batch, 1, n_mels, time_steps]
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1) # Shape: [Batch, 128]

        x = self.fc1(x)  # Output raw logits: [Batch, 1]
        return x
