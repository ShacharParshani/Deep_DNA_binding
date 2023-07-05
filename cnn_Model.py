from torch import nn
import torch.nn.functional as F
import torch


class Our_Model(nn.Module):
    def __init__(self, batch_size):
        super(Our_Model, self).__init__()
        self.batch_size = batch_size
        # The input goes into a 1D-convolutional layer of 512 kernels of width 8 and stride 1.
        self.conv1 = nn.Conv2d(1, 512, kernel_size=(1, 8), stride=1)
        # create a max pool layer pool-size 3 and stride 3
        self.mp = nn.MaxPool2d(kernel_size=4, stride=1)
        # create 3 fully connected layers with 64, 32, 32 neurons
        self.fc1 = nn.Linear(15872, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        # output layer of 6 neurons
        self.fc4 = nn.Linear(32, 6)

    def forward(self, x):
        x = x.reshape(self.batch_size, 1, 4, 41)
        x = self.conv1(x)
        x = self.mp(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x


