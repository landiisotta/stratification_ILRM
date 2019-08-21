import torch
import os

# dataset filenames
dt_files = {'ehr-file': 'cohort-new-ehrseq.csv',
            'ehr-file-test': 'cohort_test-new-ehrseq.csv',
            'vocab': 'cohort-new-vocab.csv'}

f_dtype = ['vitals',
           'encounter']

# data pre-processing parameters
data_preproc_param = {'min_diagn': 3,
                      'age_step': 15,
                      'min_seq_len': 3,
                      'max_seq_len':5000}

# model parameters
model_param = {'num_epochs': 10,
               'batch_size': 128,
               'embedding_size': 100,
               'kernel_size': 5,
               'learning_rate': 0.0001,
               'weight_decay': 1e-5
               }

# embeddings to evaluate
ev_model = ['convae', 'raw', 'svd', 'tfidf']

HCpar = {'linkage_clu': 'ward',
         'affinity_clu': 'euclidean',
         'min_cl': 3,
         'max_cl': 15}

FRpar = {'n_terms': 50}

# diseases to consider for internal validation
val_disease = ['T2D', 'PD', 'AD', 'MM', 'BC', 'PC']
select_terms = ['icd9']

# length of padded sub-sequences
len_padded = 32
dim_baseline = 100

# n iter outer clustering inspection
n_iter = 100

# save the best model
def save_best_model(epoch, model, optimizer, loss,  outdir):
    torch.save({'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, os.path.join(outdir, 'best_model.pt'))
