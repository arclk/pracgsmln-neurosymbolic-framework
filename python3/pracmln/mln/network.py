import torch
import torch.nn as nn
import torch.nn.functional as F


class Standard_Formula(nn.Module):
    def __init__(self):
        super(Standard_Formula, self).__init__()
        self.fc = nn.Linear(1, 1, bias=False)

    def forward(self, x=torch.ones(1)):
        return self.fc(x)


class Network(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.fc(x)


class MNIST_Network(nn.Module):
    def __init__(self, input_dim):
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


class RNN_Network(nn.Module):
    def __init__(self, encoding_dim=128):
        super(RNN_Network, self).__init__()
        self.claim_embedding = nn.Embedding(5089, 50, padding_idx=1)
        self.evidence_embedding = nn.Embedding(7498, 50, padding_idx=1)

        self.claim_rnn = nn.LSTM(50, encoding_dim)
        self.evidence_rnn = nn.LSTM(50, encoding_dim)

        self.fc_out = nn.Linear(encoding_dim*2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, claim, evidence):
        embedded_c = self.claim_embedding(claim)
        embedded_e = self.evidence_embedding(evidence)

        _, (outputs_c, _) = self.claim_rnn(embedded_c)
        _, (outputs_e, _) = self.evidence_rnn(embedded_e) 
        outputs_c = outputs_c[0]
        outputs_e = outputs_e[0]

        # print(outputs_e.shape)
        output = torch.cat((outputs_c, outputs_e), 1)
        output = self.fc_out(output)
        # output = self.activation(output)

        return output


    def init(self):
        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.1)

        # Init weights
        # self.apply(init_weights)

        # Initialize the embedding matrix with the one previously calculated
        self.claim_embedding.weight.data.copy_(torch.Tensor(torch.load('claim_mat.pt')))
        self.evidence_embedding.weight.data.copy_(torch.Tensor(torch.load('evidence_mat.pt')))

        # Make the weights of the embedding matrix non trainable
        self.claim_embedding.weight.requires_grad = False
        self.evidence_embedding.weight.requires_grad = False

        # Initialize the PAD vector with zeros
        self.claim_embedding.weight.data[1] = torch.zeros(50)
        self.evidence_embedding.weight.data[1] = torch.zeros(50)


class SA_Network(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SA_Network, self).__init__()
        self.embedding = nn.Embedding(25214, 50, padding_idx=1)

        self.rnn = nn.LSTM(50, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)

        output, (hidden, cell) = self.rnn(embedded)
        
        # print(outputs_e.shape)
        output = self.fc_out(hidden)
        # output = self.activation(output)

        return output


    def init(self):
        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.1)

        # Init weights
        # self.apply(init_weights)

        # Initialize the embedding matrix with the one previously calculated
        self.embedding.weight.data.copy_(torch.Tensor(torch.load('mat.pt')))
        
        # Make the weights of the embedding matrix non trainable
        self.embedding.weight.requires_grad = False
        
        # Initialize the PAD vector with zeros
        self.embedding.weight.data[1] = torch.zeros(50)
        