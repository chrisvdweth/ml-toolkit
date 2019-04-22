import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpClassifier(nn.Module):

    def __init__(self, hidden_dims, label_size, device, dropout=0.0, linear_activation='relu'):
        super(MlpClassifier, self).__init__()
        self.hidden_dims = hidden_dims
        self.linear_dims = self.hidden_dims
        self.linear_dims.append(label_size)
        self.dropout = dropout
        self.device = device

        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(1, len(self.linear_dims)):
            if dropout > 0.0:
                self.linears.append(nn.Dropout(p=dropout))
            self.linears.append(nn.Linear(self.linear_dims[i - 1], self.linear_dims[i]))
            if i == len(self.linear_dims) - 1:
                break  # no activation after output layer!!!
            if linear_activation == 'relu':
                self.linears.append(nn.ReLU())
            elif linear_activation == 'sigmoid':
                self.linears.append(nn.Sigmoid())
            elif linear_activation == 'tanh':
                self.linears.append(nn.Tanh())
            else:
                self.linears.append(nn.ReLU())


    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False


    def forward(self, batch):
        x = batch
        # Go through all layers (dropout, fully connected + activation function)
        for l in self.linears:
            x = l(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs


(log_probs)