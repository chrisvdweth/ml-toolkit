import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnType:
    GRU = 1
    LSTM = 2

class AttentionModel:
    NONE = 0
    DOT = 1
    GENERAL = 2
    CONCAT = 3

class Parameters:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))



class Attention(nn.Module):
    def __init__(self, device, method, hidden_size):
        super(Attention, self).__init__()
        self.device = device

        self.method = method
        self.hidden_size = hidden_size

        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.method == AttentionModel.GENERAL:
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == AttentionModel.CONCAT:
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = torch.FloatTensor(1, hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (batch_size, seq_len, hidden_size)
        # final_hidden_state.shape: (batch_size, hidden_size)
        # NOTE: hidden_size may also reflect bidirectional hidden states (hidden_size = num_directions * hidden_dim)
        batch_size, seq_len, _ = rnn_outputs.shape
        if self.method == AttentionModel.DOT:
            attn_weights = torch.bmm(rnn_outputs, final_hidden_state.unsqueeze(2))
        elif self.method == AttentionModel.GENERAL:
            attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
            attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
        #elif self.method == AttentionModel.CONCAT:
        #    NOT IMPLEMENTED
        else:
            raise Exception("[Error] Unknown AttentionModel.")

        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)

        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)

        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))

        return attn_hidden, attn_weights









class RnnClassifier(nn.Module):
    def __init__(self, device, params):
        super(RnnClassifier, self).__init__()
        self.params = params
        self.device = device

        # Embedding layer
        self.word_embeddings = nn.Embedding(self.params.vocab_size, self.params.embed_dim)

        # Calculate number of directions
        self.num_directions = 2 if self.params.bidirectional == True else 1

        self.linear_dims = [self.params.rnn_hidden_dim * self.num_directions] + self.params.linear_dims
        self.linear_dims.append(self.params.label_size)

        # RNN layer
        rnn = None
        if self.params.rnn_type == RnnType.GRU:
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RnnType. Currently supported: RnnType.GRU=1, RnnType.LSTM=2")
        self.rnn = rnn(self.params.embed_dim,
                       self.params.rnn_hidden_dim,
                       num_layers=self.params.num_layers,
                       bidirectional=self.params.bidirectional,
                       dropout=self.params.dropout,
                       batch_first=False)


        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_dims)-1):
            if self.params.dropout > 0.0:
                self.linears.append(nn.Dropout(p=self.params.dropout))
            linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i+1])
            self.init_weights(linear_layer)
            self.linears.append(linear_layer)
            if i == len(self.linear_dims) - 1:
                break  # no activation after output layer!!!
            self.linears.append(nn.ReLU())

        self.hidden = None

        # Choose attention model
        if self.params.attention_model != AttentionModel.NONE:
            self.attn = Attention(self.device, self.params.attention_model, self.params.rnn_hidden_dim * self.num_directions)



    def init_hidden(self, batch_size):
        if self.params.rnn_type == RnnType.GRU:
            return torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device)
        elif self.params.rnn_type == RnnType.LSTM:
            return (torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False


    def forward(self, inputs):
        batch_size, seq_len = inputs.shape

        # Push through embedding layer
        X = self.word_embeddings(inputs).transpose(0, 1)

        # Push through RNN layer
        rnn_output, self.hidden = self.rnn(X, self.hidden)

        # Extract last hidden state
        if self.params.rnn_type == RnnType.GRU:
            final_state = self.hidden.view(self.params.num_layers, self.num_directions, batch_size, self.params.rnn_hidden_dim)[-1]
        elif self.params.rnn_type == RnnType.LSTM:
            final_state = self.hidden[0].view(self.params.num_layers, self.num_directions, batch_size, self.params.rnn_hidden_dim)[-1]
        # Handle directions
        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            # final_hidden_state = h_1 + h_2               # Add both states (requires changes to the input size of first linear layer + attention layer)
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # Push through attention layer
        attn_weights = None
        if self.params.attention_model != AttentionModel.NONE:
            rnn_output = rnn_output.permute(1, 0, 2)  #
            X, attn_weights = self.attn(rnn_output, final_hidden_state)
        else:
            X = final_hidden_state

        # Push through linear layers
        for l in self.linears:
            X = l(X)

        log_probs = F.log_softmax(X, dim=1)

        return log_probs, attn_weights


    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            print("Initialize layer with nn.init.xavier_uniform_: {}".format(layer))
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

