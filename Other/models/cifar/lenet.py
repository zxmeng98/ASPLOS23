"""LeNet for CIFAR10/100

Reference:
LeCun, Yann, et al. Squeeze-and-Excitation Networks (Proceedings of the IEEE, 1998)
"""

import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, dataset):
        super(LeNet, self).__init__()

        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        else:
            raise ValueError("Incorrect Dataset Input.")

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def lenet(Dataset):
    return LeNet(Dataset)
