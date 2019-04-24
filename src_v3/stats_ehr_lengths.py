'''
This code summarizes the length of the ehrs and drops the sequences (considering all possible terms) 
shorter than len_min (in utils.data_preprocessing_pars). Patients to drop are saved in list_mrnToDrop.csv
'''
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import data_preprocessing_pars

def stats_ehr_lengths(outdir):
    file_name = 'ehr-shuffle.csv'

    with open(os.path.join(outdir, file_name)) as f:
        rd = csv.reader(f)
        ehr_shuffle = {}
        sub_len = []
        for r in rd:
            sub_len.append(len(r[1::]))
            ehr_shuffle.setdefault(r[0], list()).extend(r[1::])
    print("The average number of tokens for each time slot of {0} days is {1:.2f}".format(data_preprocessing_pars['age_step'],
                                                                                          np.mean(sub_len)))

    plt.figure(figsize=[20,10])
    plt.xticks(np.arange(0, max(sub_len), 1))
    plt.hist(sub_len)
    plt.savefig(os.path.join(outdir, 'hist-encounter_seqlengths.png'))

    l = []
    for mrn in ehr_shuffle:
        l.append(len(ehr_shuffle[mrn]))
    print("The average length of ehr sequences is: {0:.2f}".format(np.mean(l)))
    count = 0
    for ll in l:
        if ll<data_preprocessing_pars['len_min']:
            count += 1
    print("{0} of {1} patients have less than {2} records".format(count, len(l),
                                                                  data_preprocessing_pars['len_min']))
    print("The sequence length ranges from {0} to {1}".format(min(l), max(l)))

    plt.figure(figsize=[20,10])
    plt.xticks(np.arange(0, max(l), 100))
    plt.hist(l, bins = 36)
    plt.savefig(os.path.join(outdir, 'hist-seq_lengths.png'))

    with open(os.path.join(outdir, 'list_mrnToDrop.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',')
        discard_list = []
        for mrn in ehr_shuffle:
            if len(ehr_shuffle[mrn]) < data_preprocessing_pars['len_min']:
                discard_list.append(mrn)
        wr.writerow(discard_list)

    print("We are dropping {0} out of {1} patients".format(len(discard_list), len(ehr_shuffle)))
