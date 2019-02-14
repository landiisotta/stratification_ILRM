"""
Define the data to feed the deep learning model.

If batch_size = 1, each sequence is padded to reach length multiple of
"padded_seq_len"; each sequence is tehn trimmed in subsequences of
length "padded_seq_len".
"""

from torch.utils.data import Dataset
from utils import padded_seq_len
import torch
import os
import csv

"""
EHR data class
"""


class EHRdata(Dataset):

    def __init__(self, datadir, ehr_file):
        self.ehr = {}
        with open(os.path.join(datadir, ehr_file)) as f:
            rd = csv.reader(f)
            for r in rd:
                seq = list(map(int, r[1::]))

                if len(seq) < padded_seq_len:
                    self.ehr[r[0]] = seq + [0] * \
                        (padded_seq_len - len(seq))

                elif len(seq) % padded_seq_len != 0:
                    self.ehr[r[0]] = seq + [0] * \
                        (padded_seq_len - len(seq) % padded_seq_len)

                else:
                    self.ehr[r[0]] = seq


    def __getitem__(self, index):
        ehr_list = [[mrn, term] for mrn, term in self.ehr.items()]
        seq = ehr_list[index][1]
        pat = ehr_list[index][0]
        return seq, pat


    def __len__(self):
        return len(self.ehr)


def ehr_collate(batch):
    data = []
    mrn = []
    for seq, pat in batch:
        mrn.append(pat)

        if len(seq) == padded_seq_len:
            data.append(torch.tensor(
                [seq], dtype=torch.long).view(-1, padded_seq_len))

        elif len(seq) % padded_seq_len == 0:
            sq = []
            for idx in range(0, len(seq) - padded_seq_len + 1, padded_seq_len):
                sq.append(seq[idx:idx + padded_seq_len])
            data.append(torch.tensor(
                sq, dtype=torch.long).view(-1, padded_seq_len))

        else:
            raise Warning(
                'Not all the sequences have length multiple than {0}'.format(
                    padded_seq_len))

    return [data, mrn]
