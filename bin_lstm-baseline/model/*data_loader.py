'''
Load padded sequences for LSTM baseline
'''
import torch
from torch.utils.data import Dataset
import os
import csv
import utils

##define myData class
class myData(Dataset):
    def __init__(self, data_folder, list_mrn_file, trimmed_ehr_file, labels):
        with open(os.path.join(data_folder, list_mrn_file)) as f:
            rd = csv.reader(f)
            self.list_mrn = [r for r in rd]
        with open(os.path.join(data_folder, trimmed_ehr_file)) as f:
            rd = csv.reader(f)
            self.ehr = [list(map(int, r)) for r in rd]    
        with open(os.path.join(data_folder, labels)) as f:
            rd = csv.reader(f)
            self.labels = [r[0] for r in rd]
                    
    def __getitem__(self, index):
        seq = self.ehr[index]
        pat = self.list_mrn[index]
        lab = self.labels[index]
        return seq, pat, lab

    ##len(dataset) returns the number of patients     
    def __len__(self):
        return len(self.list_mrn)

def my_collate(batch):
    data = []
    mrn = []
    lab = []
    lengths = []
    for seq, pat, l in batch:
        data.append(seq)
        mrn.append(pat)
        lab.append(l)
    data = torch.tensor(data, dtype=torch.long)
    lengths = [int(sum((d>=0).float()).item()) for d in data]
    return [data, mrn, lengths, lab]
        
