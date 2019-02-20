import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils

class LSTMehrEncoding(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, batch_size):
        super(LSTMehrEncoding, self).__init__()
        
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.emb_dim = emb_dim

        self.n_layers = 2
        self.n_lstm_units = int(self.emb_dim / 2)
        self.ch_l2 = int(self.n_lstm_units / 2)
        self.ch_l3 = int(self.ch_l2/2)

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.emb_dim, self.n_lstm_units, self.n_layers, batch_first=True, dropout=0.5)
        self.linear1 = nn.Linear(self.n_lstm_units, self.ch_l2)
        self.linear2 = nn.Linear(self.ch_l2, self.ch_l3)
        self.linear3 = nn.Linear(self.ch_l3, vocab_size)

    def init_hidden(self):
        hidden_a = torch.randn(self.n_layers, self.batch_size, self.n_lstm_units)
        hidden_b = torch.randn(self.n_layers, self.batch_size, self.n_lstm_units)
        
        hidden_a = hidden_a.cuda()
        hidden_b = hidden_b.cuda()
        
        return (hidden_a, hidden_b)

    def forward(self, x, x_lengths):
        self.hidden = self.init_hidden()
        input_vect = x
        seq_len = input_vect.shape[1]
        embeds = self.embedding(x)
        ##we make sure the LSTM won't see the padding
        out = nn.utils.rnn.pack_padded_sequence(embeds, x_lengths, batch_first=True)
        #print(out.size())
        out, self.hidden = self.lstm(out, self.hidden)
        #print(out.shape)
        out, h = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.contiguous().view(-1, out.shape[2])
        out = self.linear1(out)
        out = F.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        encoded_vect = out.view(self.batch_size, seq_len * self.ch_l3)
        out = self.linear3(out)
        #mask = torch.tensor([[1]*seq_len+[0]*(self.max_seq_len-seq_len)]*self.vocab_size)
        #print(mask.shape)
        #mask = mask.view(-1, self.vocab_size, self.max_seq_len)
        out = out.view(self.batch_size, self.vocab_size, seq_len)
        #print(mask.shape)
        #out = out * mask

        return(out, encoded_vect)


def accuracy(out, target):
    logsoft = F.log_softmax(out, dim=1)
    pred = torch.argmax(logsoft, dim=1)
    mask = (target >= 0).float()
    nb_tokens = int(torch.sum(mask).item())
    return torch.sum((pred==target).float() * mask)/float(nb_tokens)

criterion = nn.CrossEntropyLoss()

metrics = {'accuracy': accuracy}
