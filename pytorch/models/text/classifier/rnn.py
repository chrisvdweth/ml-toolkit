import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnClassifier(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, rnn_hidden_dim, linear_dims, label_size,
                 num_layers=1, bidirectional=False, dropout=0.0, linear_activation='relu', device='cpu'):
        super(RnnClassifier, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        # Embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Recurrent layer (GRU)
        self.directions_count = 2 if bidirectional == True else 1

        self.linear_dims = [self.rnn_hidden_dim * self.directions_count] + linear_dims
        self.linear_dims.append(label_size)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, self.rnn_hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, self.rnn_hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_dims)-1):
            if dropout > 0.0:
                self.linears.append(nn.Dropout(p=dropout))
            linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i+1])
            self.init_weights(linear_layer)
            self.linears.append(linear_layer)
            if i == len(self.linear_dims) - 1:
                break  # no activation after output layer!!!
            if linear_activation.lower() == 'relu':
                self.linears.append(nn.ReLU())
            elif linear_activation.lower() == 'tanh':
                self.linears.append(nn.Tanh())
            elif linear_activation.lower() == 'sigmoid':
                self.linears.append(nn.Sigmoid())
            elif linear_activation.lower() == 'logsigmoid':
                self.linears.append(nn.LogSigmoid())
            else:
                self.linears.append(nn.ReLU())
        self.hidden = None


    def init_hidden(self, batch_size):
        if self.rnn_type == 'gru':
            return torch.zeros(self.num_layers * self.directions_count, batch_size, self.rnn_hidden_dim).to(self.device)
        elif self.rnn_type == 'lstm':
            return (torch.zeros(self.num_layers * self.directions_count, batch_size, self.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.num_layers * self.directions_count, batch_size, self.rnn_hidden_dim).to(self.device))
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False


    def forward(self, X_sorted, X_length_sorted, method='last_step'):
        # Push through embedding layer
        X = self.word_embeddings(X_sorted)
        # Transpose (batch_size, seq_len, dim) to (seq_len, batch_size, dim)
        X = torch.transpose(X, 0, 1)
        # Pack padded sequence
        X = nn.utils.rnn.pack_padded_sequence(X, X_length_sorted)
        # Push through RNN layer
        X, self.hidden = self.rnn(X, self.hidden)
        # Unpack packed sequence (no longer needed since we use the hidden state)
        #X, output_lengths = nn.utils.rnn.pad_packed_sequence(X)

        if True:
            if self.rnn_type == 'gru':
                final_state = self.hidden.view(self.num_layers, self.directions_count, X_sorted.shape[0], self.rnn_hidden_dim)[-1]
            elif self.rnn_type == 'lstm':
                final_state = self.hidden[0].view(self.num_layers, self.directions_count, X_sorted.shape[0], self.rnn_hidden_dim)[-1]
            else:
                raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

            if self.directions_count == 1:
                X = final_state.squeeze()
            elif self.directions_count == 2:
                h_1, h_2 = final_state[0], final_state[1]
                #X = h_1 + h_2                # Add both states (requires changes to the input size of first linear layer)
                X = torch.cat((h_1, h_2), 1)  # Concatenate both states
        else:
            X = X[-1]

        for l in self.linears:
            X = l(X)
        log_probs = F.log_softmax(X, dim=1)
        return log_probs


    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            print("Initiliaze layer with nn.init.xavier_uniform_: {}".format(layer))
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

