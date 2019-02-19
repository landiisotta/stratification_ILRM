"""
Define the data to feed the deep learning model.

If batch_size = 1, each sequence is padded to reach length multiple of
"padded_seq_len"; each sequence is tehn trimmed in subsequences of
length "padded_seq_len".
"""

from torch.utils.data import Dataset
from utils import len_padded
import torch
import os
import csv

"""
EHR data class
"""




class EHRdata(Dataset):

    def __init__(self, datadir, ehr_file):
        self.ehr_list = []
        with open(os.path.join(datadir, ehr_file)) as f:
            rd = csv.reader(f)
            for r in rd:
                mrn = r[0]
                seq = list(map(int, r[1::]))

                if len(seq) < len_padded:
                    ps = [seq + [0] * (len_padded - len(seq))]

                elif len(seq) % len_padded != 0:
                    seq += [0] * (len_padded - len(seq) % len_padded)
                    ps = []
                    for i in range(0, len(ps) - len_padded + 1, len_padded):
                        ps.append(seq[i:i + len_padded])

                else:
                    ps = [seq]
        self.ehr_list.append([mrn, ps])


    def __getitem__(self, index):
        seq = self.ehr_list[index][1]
        pat = self.ehr_list[index][0]
        return (seq, pat)


    def __len__(self):
        return len(self.ehr_list)



def ehr_collate(batch):
    data = []
    mrn = []
    for seq, pat in batch:
        mrn.append(pat)
        data.append(torch.tensor(
            seq, dtype=torch.long).view(-1, len_padded))
    return [data, mrn]
