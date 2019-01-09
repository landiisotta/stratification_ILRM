"""
Run the data preprocessing steps. Change parameters in utils.py
"""

from create_ehr_cohorts import create_ehr_cohorts
from filter_collection_frequencies import filter_collection_frequencies
from preprocessing_ehr import preprocessing_ehr
from stats_ehr_lengths import stats_ehr_lengths
from data_preparation import data_preparation

outdir = create_ehr_cohorts()    
filter_collection_frequencies(outdir)
preprocessing_ehr(outdir)
stats_ehr_lengths(outdir)
data_preparation(outdir)

"""
Save the output data directory path to a txt file in the ../bin 
directory. It is used to load data for the DL model. It changes at every 
run of the data preprocessing code. 
"""
f = open("outdir.txt", "w")
f.write(outdir)
f.close()
