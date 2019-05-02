from os import path
import h5py
import csv
from utils import f_dtype

"""
Load the data as a dictionary with MRNs as keys. Every element (patient) of
the dictonary is then a sorted list of tuples in the form:

[CONCEPT_CODE, AGE_IN_DAYS, YEAR, DAY]

You can map the CONCEPT_CODE to the corresponding label using the vocabulary
index (i.e., "vidx").
"""


def load_dataset(datadir):
    # load data index
    vidx = _load_index(path.join(datadir, 'vocab-idx.csv'),
                       'clinical concepts')
    pidx = _load_index(path.join(datadir, 'patient-idx.csv'), 'patient MRNs')
    
    # load patient data
    pehrs = {}
    with h5py.File(path.join(datadir, 'pt-history.hdf5'), 'r') as f:
        for p in f:
            
            pi = int(p)
            mrn = pidx[pi]

            pehrs[mrn] = {}

            # demography information
            demog = {}
            for a in f[p].attrs:
                demog[int(a)] = int(f[p].attrs[a])
            pehrs[mrn]['demog'] = demog

            # clinical events
            hst = list(zip(f[p]['event'][:],
                            f[p]['age_in_days'][:],
                            f[p]['year'][:],
                            f[p]['day'][:]))

            pehrs[mrn]['events'] = sorted(set(hst), key=lambda x: (x[1]))

            if len(pehrs) == 100:
                break

    print('\nLoaded data for %d patients\n' % len(pehrs))
    return (pehrs, vidx, pidx)


def _load_index(fidx, label):
    with open(fidx, encoding='utf-8') as f:
        d = csv.reader(f)
        next(d)
        idx = {int(r[1]): r[0] for r in d}
    print('Loaded %d %s' % (len(idx), label))
    return idx


#load_dataset('..')
