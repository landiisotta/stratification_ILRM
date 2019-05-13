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
                      'max_seq_len':10000}

# model parameters
model_param = {'num_epochs': 5,
               'batch_size': 1,
               'embedding_size': 100,
               'kernel_size': 5,
               'learning_rate': 0.0001,
               'weight_decay': 1e-5
               }

# embeddings to evaluate
ev_model = ['conv-ae', 'raw', 'svd', 'dp']
# list of diseases to consider
diseases = []

# length of padded sub-sequences
len_padded = 64
dim_baseline = 100

# save the best model
def save_best_model(epoch, model, optimizer, loss,  outdir):
    torch.save({'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, os.path.join(outdir, 'best_model.pt'))
