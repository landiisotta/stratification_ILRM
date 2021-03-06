import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils

class ehrEncoding(nn.Module):
    
    def __init__(self, vocab_size, max_seq_len, emb_dim, kernel_size):
        super(ehrEncoding, self).__init__()
        
        ##variables
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.kernel_size = kernel_size
        self.ch_l1 = int(emb_dim / 2)
        #self.ch_l2 = int(self.ch_l1 / 2)
        self.padding = int((kernel_size - 1) / 2)
        self.features = math.floor(max_seq_len + 2*self.padding - kernel_size + 1) + 2*self.padding - kernel_size + 1
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cnn_l1 = nn.Conv1d(emb_dim, self.ch_l1, kernel_size=kernel_size, padding=self.padding)
        #self.cnn_l2 = nn.Conv1d(self.ch_l1, self.ch_l2, kernel_size=kernel_size, padding=self.padding)
        self.linear1 = nn.Linear(self.ch_l1*self.features, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, vocab_size*max_seq_len)

        #self.ch_l2 = int(self.ch_l1 / 2)
        #self.cnn_l2 = nn.Conv1d(self.ch_l1, self.ch_l2, kernel_size=kernel_size, padding=self.padding)
        #self.bn1 = nn.BatchNorm1d(self.ch_l1)
        #self.bn2 = nn.BatchNorm1d(self.ch_l2)

    def forward(self, x):
        input_vect = x
        
        ##embedding
        embeds = self.embedding(x)
        embeds = embeds.permute(0,2,1)
        
        ##one layer of cnn (no batch normalization, cause batch_size is 1 and dropout is used)
        out = F.relu(self.cnn_l1(embeds))
        out = F.max_pool1d(out, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        #out = F.relu(self.cnn_l2(out))
        #out = F.max_pool1d(out, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        out = out.view(-1, out.shape[2]*out.shape[1])
        
        ##2 layers of encoding, two of decoding
        out = self.linear1(out)
        out = F.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        #out = F.dropout(out)
        #out = F.relu(out)
        
        encoded_vect = out.view(-1,out.shape[1])
        
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = out.view(-1, self.vocab_size, input_vect.shape[1])
 
        return(out, encoded_vect)


def accuracy(out, target):
    logsoft = F.log_softmax(out, dim=1)
    pred = torch.argmax(logsoft, dim=1)
    return torch.sum((pred==target).float())/(out.shape[2]*out.shape[0])

criterion = nn.CrossEntropyLoss()

metrics = {'accuracy': accuracy}
