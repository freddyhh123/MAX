import torch.nn as nn
import torch.nn.functional as F

class topGenreClassifer(nn.Module):
    def __init__(self):
        super(topGenreClassifer, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (input_size // 4), 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (input_size // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x