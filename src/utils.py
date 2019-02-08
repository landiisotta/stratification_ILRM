import torch
import os

##PATHS
disease_folder = 'parkinson'
data_folder = os.path.expanduser('~/data1/stratification_ILRM/data/%s/' % disease_folder) 

ehr_file = 'cohort-new_ehr.csv'
mt_to_ix_file = 'cohort-new_vocab.csv'

##DISEASES AND TERM TYPES
#diseases = ["alzheimer's disease", "multiple myeloma", "parkinson's disease"]
diseases = ["alzheimer's disease", "multiple myeloma", "parkinson's disease", "malignant neopasm of female breast", "malignant tumor of prostate"]
dtype = ['icd9', 'icd10', 'medication']

##PREPROCESSING STEP PARAMETERS
data_preprocessing_pars = {'min_diagn' : 3,
                           'n_rndm' : 0,
                           'age_step' : 15,
                           'len_min' : 3}
##MODEL PARAMETERS
model_pars = {'num_epochs' : 100,
              'batch_size' : 8, ##batch size
              'embedding_dim' : 128,
              'kernel_size' : 5,
              'learning_rate' : 0.0001}

##LENGTH OF PADDED SUBSEQUENCES
L = 64

##SAVE BEST MODEL
def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
