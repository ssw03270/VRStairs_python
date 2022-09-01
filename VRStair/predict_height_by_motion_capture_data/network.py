import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(70, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        x = nn.functional.elu(self.fc1(x))
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.elu(self.fc3(x))
        x = nn.functional.elu(self.fc4(x))

        return x


# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.conv1 = nn.Conv1d(10, 64, kernel_size=1)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
#         self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
#
#         self.conv4 = nn.Conv1d(256, 128, kernel_size=1)
#         self.conv5 = nn.Conv1d(128, 64, kernel_size=1)
#         self.conv6 = nn.Conv1d(64, 1, kernel_size=1)
#
#         self.fc = nn.Linear(6, 3)
#
#     def forward(self, x):
#         input_size = x.shape[0]
#         x = nn.functional.elu(self.conv1(x))
#         x = nn.functional.elu(self.conv2(x))
#         x = nn.functional.elu(self.conv3(x))
#         x = nn.functional.elu(self.conv4(x))
#         x = nn.functional.elu(self.conv5(x))
#         x = nn.functional.elu(self.conv6(x))
#
#         x = x.reshape(input_size, -1)
#         x = nn.functional.elu(self.fc(x))
#
#         return x
