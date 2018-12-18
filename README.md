## Create virtual environment with ```python3```

Move to the ```stratification_ILRM``` folder:

```
mkdir myvenv
pip install virtualenv
virtualenv -p python 3 myvenv 
```

Activate/deactivate the virtual environment:
```
source myvenv/bin/activate

deactivate
```
## Install dependencies

```
pip install jupyterlab
```
Run it with ```jupyter-lab``` from command line.

```
pip install matplotlib
pip install torch
pip install numpy
pip install sklearn
```

## Modify and run the code
In ```stratificattion_ILRM``` folder:

### Modify

> Create a folder name_folder in /data/ where to save .csv files. 

> Modify /bin/utils.py

- ```disease_folder``` = name_folder
- ```diseases``` = list of the diseases to consider
- ```dtype``` = list of the medical terms to consider among _icd9/10, medication, lab, vitals, cpt_
- ```data_preprocessing_pars``` = dictionary of parameters: ```min_diagn```: include only patients with minimum number of diagnoses,
```n_rndm```: number of random patients to include in the dataset, ```age_step```: number of days that represent an encounter (unique, shuffled terms),
```len_min```: minimum length of ehr sequences after preprocessing
- ```model_pars``` = dictionary with architecture parameters: ```num_epochs```: number of epochs, ```batch_size```: **set batch size equal 1 for the CNN-AE architecture**, ```embedding_dim```: dimension of the embedding of medical terms,
```kernel_size```: size of the CNN kernel, ```learning_rate```: learning rate.
- ```L```= length of the subsequences to dicide the ehr sequences in

### Run in sequence
```
python /bin/run_preprocessing.py
python /bin/main.py
python /bin_baseline-lstm/main.py
```

> Open jupyter-lab and run /bin/data_clustering_visualization.ipynb

Change minimum number/maximum number of clusters in the ```hclust_ehr()``` function, select the number of the most frequent terms and the type of the terms to display for each subcluster in ```freq_term()``` function.
