import torch
import os

# dataset filenames
dt_files = {'ehr': 'cohort-new_ehr.csv',
            'vocab': 'cohort-new_vocab.csv'}

# diseases and term types

# diseases = ['alzheimer\'s disease',
#            'multiple myeloma',
#            'parkinson\'s disease']

diseases = ['alzheimer\'s disease',
            'multiple myeloma',
            'parkinson\'s disease',
            'malignant neopasm of female breast',
            'malignant tumor of prostate']

dtype = ['icd9',
         'icd10',
         'medication']

# data pre-processing parameters
data_preproc_param = {'min_diagn': 3,
                      'n_rndm': 0,
                      'age_step': 15,
                      'len_min': 3}

# model parameters
model_param = {'num_epochs': 20,
               'batch_size': 32,
               'embedding_size': 100,
               'kernel_size': 5,
               'learning_rate': 0.001,
               'weight_decay': 1e-5
               }

# length of padded sub-sequences
len_padded = 96


# save the best model


def save_best_model(state, outdir):
    torch.save(state, os.path.join(outdir, 'best_model.pt'))
