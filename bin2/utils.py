import torch
import os

##PATHS
disease_folder = 'mixed_rid'
data_folder = os.path.expanduser('~/data1/stratification_ILRM/data/%s/' % disease_folder) 

ehr_file = 'cohort-new_ehr.csv'
mt_to_ix_file = 'cohort-new_vocab.csv'

##DISEASES AND TERM TYPES
diseases = ["malignant neoplasm of female breast", "malignant tumor of prostate", "alzheimer's disease", "multiple myeloma", "parkinson's disease"]
dtype = ['icd9', 'icd10', 'medication']

##PREPROCESSING STEP PARAMETERS
data_preprocessing_pars = {'min_diagn' : 3,
                           'n_rndm' : 3000,
                           'age_step' : 15,
                           'len_min' : 3}
##MODEL PARAMETERS
model_pars = {'num_epochs' : 100,
              'batch_size' : 1, ##batch size equal 1 required
              'embedding_dim' : 128,
              'kernel_size' : 5,
              'learning_rate' : 0.001}

##LENGTH OF PADDED SUBSEQUENCES
L = 256

##SAVE BEST MODEL
def save_best_model(state, folder):
    print("-- Found new best")
    torch.save(state, os.path.join(folder, "best_model.pt"))
