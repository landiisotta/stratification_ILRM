import torch
import os

##PATHS
disease_folder = 'neurodev'
data_folder = os.path.expanduser('~/data1/stratification_ILRM/data/%s/' % disease_folder) 

ehr_file = 'cohort-new_ehr.csv'
mt_to_ix_file = 'cohort-new_vocab.csv'

##DISEASES AND TERM TYPES
diseases = ['autism', 'autistic', 'pervasive developmental disorder', 'asperger', 'mental retardation', 'attention deficit']
#diseases = ["alzheimer's disease", "multiple myeloma", "parkinson's disease"]
#diseases = ["alzheimer's disease", "multiple myeloma", "parkinson's disease", "malignant neoplasm of female breast", "malignant tumor of prostate"]
#diseases = ["diabetes", "alzheimer's disease", "multiple myeloma", "parkinson's disease", "malignant neoplasm of female breast", "malignant tumor of prostate"]
dtype = ['icd9', 'icd10', 'medication', 'lab', 'cpt', 'procedure']
#diseases = ["multiple myeloma", "autism"]
#dtype = ['icd9', 'icd10']

##PREPROCESSING STEP PARAMETERS
data_preprocessing_pars = {'min_diagn' : 3,
                           'n_rndm' : 10000,
                           'age_step' : 15,
                           'len_min' : 3}
##MODEL PARAMETERS
model_pars = {'num_epochs' : 50,
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
