import torch
import torch.nn as nn
import torch.nn.functional as F


class Standard_Formula(nn.Module):
    def __init__(self):
        super(Standard_Formula, self).__init__()
        self.fc = nn.Linear(1, 1, bias=False)

    def forward(self, x=torch.ones(1)):
        return self.fc(x)


class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class ABSTRCT_TypeNetwork(nn.Module):
    def __init__(self):
        super(ABSTRCT_TypeNetwork, self).__init__()
        self.fc1 = nn.Linear(25, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)
        
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()

    def forward(self, input):
        output = self.activation(self.fc1(input))
        output = self.activation(self.fc2(output))
        output = self.activation(self.fc3(output))
        output = self.fc4(output)

        return output


class ABSTRCT_LinkNetwork(nn.Module):
    def __init__(self):
        super(ABSTRCT_LinkNetwork, self).__init__()
        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)
        
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()

    def forward(self, input):
        output = self.activation(self.fc1(input))
        output = self.activation(self.fc2(output))
        output = self.activation(self.fc3(output))
        output = self.fc4(output)

        return output
