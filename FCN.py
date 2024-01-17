import torch

import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        # 3 fully connected layers
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x,dim=1)
