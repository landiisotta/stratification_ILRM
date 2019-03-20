import torch
import os

# dataset filenames
dt_files = {'ehr': 'cohort-new_ehr.csv',
            'vocab': 'cohort-new_vocab.csv',
            'diseases': 'cohort-mrn_diseases.csv'}

# diseases and term types

diseases = ['alzheimer\'s disease',
            'multiple myeloma',
            'parkinson\'s disease',
            'malignant neopasm of female breast',
            'malignant tumor of prostate']
            #,'type 2 dyabetes']

dtype = ['icd9',
         'icd10',
         'medication',
         'lab',
         'procedure',
         'cpt']

# data pre-processing parameters
data_preproc_param = {'min_diagn': 3,
                      'n_rndm': 10000,
                      'age_step': 15,
                      'len_min': 3}

# model parameters
model_param = {'num_epochs': 5,
               'batch_size': 8,
               'embedding_size': 200,
               'kernel_size': 5,
               'learning_rate': 0.0001,
               'weight_decay': 1e-5
               }

# length of padded sub-sequences
len_padded = 32
seq_overlap = 0


# save the best model


def save_best_model(epoch, model, optimizer, loss,  outdir):
    torch.save({'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, os.path.join(outdir, 'best_model.pt'))
