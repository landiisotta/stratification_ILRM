'''
With batch_size=1 each sequence is padded to reach length multiple of L, 
then each sequence is trimmed in subsequences of length L.
These data are input of the DL model.
'''
import torch
from torch.utils.data import Dataset
import os
import csv
import utils

##define myData class
class myData(Dataset):
    def __init__(self, data_folder, ehr_file):
        self.ehr = {}
        with open(os.path.join(data_folder, ehr_file)) as f:
            rd = csv.reader(f)
            for r in rd:
                self.ehr[r[0]] = list(map(int, r[1::]))
    
    def __getitem__(self, index):
        ehr_list = []
        for mrn, term in self.ehr.items():
            ehr_list.append([mrn, term])
        seq = ehr_list[index][1]
        pat = ehr_list[index][0]
        
        return seq, pat
    ##len(dataset) returns the number of patients     
    def __len__(self):
        return len(self.ehr)

def my_collate(batch):
    data = []
    mrn = []
    ##keep track of the original length of the padded subsequences for LSTM baseline
    batch = sorted(batch, key=lambda x:len(x[0]), reverse=True)
    lengths = [len(s) for s, _ in batch]
    max_seq_len = max(lengths) 
    for seq, pat in batch:
        mrn.append(pat)
        if len(seq) < max_seq_len:
            data.append(seq + [0]*(max_seq_len - len(seq)))
        else:
            data.append(seq)
    data = torch.tensor(data, dtype=torch.long)
    return [data, mrn, lengths]
        
