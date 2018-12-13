import torch
import os
from datetime import datetime

##PATHS
disease_folder = 'autism'
data_folder = os.path.expanduser('~/data1/complex_disorders/data/%s/' % disease_folder)

mt_to_ix_file = 'cohort-new_vocab.csv'
ehr_file = 'cohort-new_ehr.csv'

##MODEL PARAMETERS
model_pars = {'num_epochs' : 150,
              'batch_size' : 8,
              'embedding_dim' : 128,
              'learning_rate' : 0.001}

L = 256

def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
