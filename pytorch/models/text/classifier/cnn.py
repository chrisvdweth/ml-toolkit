import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnClassifier(nn.Module):

    def __init__(self, seq_len, label_size, vocab_size, embedding_dim, conv_kernel_sizes, out_channels, maxpool_kernel_size, linear_dims, dropout_linears=0.0, activation_linears='relu'):
        super(CnnClassifier, self).__init__()
        self.label_size = label_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.conv_kernel_sizes = conv_kernel_sizes
        self.out_channels = out_channels
        self.maxpool_kernel_size = maxpool_kernel_size
        self.dropout_linears = dropout_linears
        self.linear_dims = linear_dims

        self.linear_dims.append(label_size)


        # Embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.flatten_size = 0

        self.common_conv_padding = 1
        self.common_conv_stride = 1
        self.common_maxpool_dilation = 1
        self.common_maxpool_padding = 0

        self.conv_layers = nn.ModuleDict()
        for conv_kernel_size in conv_kernel_sizes:
            self.conv_layers['conv_{}'.format(conv_kernel_size)] = nn.Conv1d(in_channels=embedding_dim,
                                                                             out_channels=out_channels,
                                                                             kernel_size=conv_kernel_size,
                                                                             stride=self.common_conv_stride,
                                                                             padding=self.common_conv_padding)
            # Calculate the length of the conv output
            conv_out_size = self._calc_conv_output_size(seq_len,
                                                        conv_kernel_size,
                                                        self.common_conv_stride,
                                                        self.common_conv_padding)
            # Calculate the length of the maxpool output
            maxpool_out_size = self._calc_maxpool_output_size(conv_out_size,
                                                              maxpool_kernel_size,
                                                              self.common_maxpool_padding,
                                                              maxpool_kernel_size,
                                                              self.common_maxpool_dilation)
            # Add all lengths together
            self.flatten_size += maxpool_out_size

        self.flatten_size *= out_channels

        self.maxpool_layers = nn.ModuleDict()
        for conv_kernel_size in conv_kernel_sizes:
            self.maxpool_layers['maxpool_{}'.format(conv_kernel_size)] = nn.MaxPool1d(kernel_size=maxpool_kernel_size)

        self.linear_dims = [self.flatten_size] + self.linear_dims

        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(1, len(self.linear_dims)):
            if dropout_linears > 0.0:
                self.linears.append(nn.Dropout(p=dropout_linears))
            self.linears.append(nn.Linear(self.linear_dims[i - 1], self.linear_dims[i]))
            if i == len(self.linear_dims) - 1:
                break  # no activation after output layer!!!
            if activation_linears == 'relu':
                self.linears.append(nn.ReLU())
            elif activation_linears == 'sigmoid':
                self.linears.append(nn.Sigmoid())
            elif activation_linears == 'tanh':
                self.linears.append(nn.Tanh())
            else:
                self.linears.append(nn.ReLU())



    def forward(self, batch):
        embeds = self.word_embeddings(batch)
        # Embedding output shape: batch size x seq_length x embedding_dimension
        # Turn (batch_size x sequence length x embedding dimension) into (batch_size x input channels x sequence length) for CNN
        # (note: embedding dimension = input channels)
        x = embeds.transpose(1, 2)
        # Conv1d input shape: batch size x input channels x input length
        all_outs = []
        for conv_kernel_size in self.conv_kernel_sizes:
            out = self.conv_layers['conv_{}'.format(conv_kernel_size)](F.relu(x))
            out = self.maxpool_layers['maxpool_{}'.format(conv_kernel_size)](out)
            out = out.view(out.size(0), -1)
            all_outs.append(out)
        # Concatenate all outputs from the different conv layers
        x = torch.cat(all_outs, 1)
        # Go through all layers (dropout, fully connected + activation function)
        for l in self.linears:
            x = l(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs


    def _calc_conv_output_size(self, seq_len, kernel_size, stride, padding):
        return int(((seq_len - kernel_size + 2*padding) / stride) + 1)

    def _calc_maxpool_output_size(self, seq_len, kernel_size, padding, stride, dilation):
        return int(math.floor( ( (seq_len + 2*padding - dilation*(kernel_size-1) - 1) / stride ) + 1 ))
