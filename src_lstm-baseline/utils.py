import torch
import os
from datetime import datetime

##PATHS
disease_folder = 'mixed'
data_folder = os.path.expanduser('~/data1/stratification_ILRM/data/%s/' % disease_folder)

mt_to_ix_file = 'cohort-new_vocab.csv'
ehr_file = 'cohort-new_ehr.csv'

##MODEL PARAMETERS
model_pars = {'num_epochs' : 5,
              'batch_size' : 4,
              'embedding_dim' : 128,
              'learning_rate' : 0.001}

def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
