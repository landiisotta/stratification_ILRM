import torch
import os

# dataset filenames
dt_files = {'ehr': 'cohort-new_ehr.csv',
            'vocab': 'cohort-new_vocab.csv',
            'diseases': 'cohort-mrn_diseases.csv'}

f_dtype = ['vitals',
           'encounter']

# data pre-processing parameters
data_preproc_param = {'min_diagn': 3,
                      'n_rndm': 10000,
                      'age_step': 15,
                      'min_seq_len': 3,
                      'max_seq_len':10000}

# model parameters
model_param = {'num_epochs': 20,
               'batch_size': 16,
               'embedding_size': 100,
               'kernel_size': 5,
               'learning_rate': 0.0001,
               'weight_decay': 1e-5
               }

# length of padded sub-sequences
len_padded = 64
seq_overlap = 0


# save the best model
def save_best_model(epoch, model, optimizer, loss,  outdir):
    torch.save({'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, os.path.join(outdir, 'best_model.pt'))
