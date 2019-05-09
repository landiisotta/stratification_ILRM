from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from scipy import stats
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from os import path
import deep_patient as dp
from time import time
from datetime import datetime
import numpy as np
import utils as ut
import sys
import argparse
import random
import csv
import os


def run_baselines(indir, outdir, n_dim=ut.dim_baseline,
                  test_set=False):

    try:
        os.mkdir(outdir)
    except OSError:
        pass
    
    if test_set:
        print('Loading test dataset')
        mrns_ts, raw_data_ts, ehrs_ts, vocab = _load_test_dataset(indir)
        _save_mrns(outdir, mrns_ts, 'bs-mrn-test.txt')
        print('Loading training dataset')
        mrns, raw_data, ehrs, _ = _load_ehr_dataset(indir)
    else:
        print('Loading training datasets')
        mrns, raw_data, ehrs, vocab = _load_ehr_dataset(indir)
        mrns_ts = mrns
        raw_data_ts = raw_data
        ehrs_ts = ehrs
 
    _save_mrns(outdir, mrns, 'bs-mrn.txt')
    
    print('\nRescaling the COUNT matrix')
    scaler = MinMaxScaler()
    raw_mtx = scaler.fit_transform(raw_data)
    _save_matrices(outdir, 'raw-mtx.npy', raw_mtx)

    print('Applying SVD')
    svd = TruncatedSVD(n_components=n_dim)
    svd_mtx = svd.fit_transform(raw_mtx)
    _save_matrices(outdir, 'svd-mtx.npy', svd_mtx)

    if test_set:
        print('\nRescaling the COUNT test matrix')
        scaler = MinMaxScaler()
        raw_mtx_ts = scaler.fit_transform(raw_data_ts)
        _save_matrices(outdir, 'raw-mtx-test.npy', raw_mtx_ts)

        print('Applying SVD to test set')
        svd_mtx_ts = svd.fit_transform(raw_mtx_ts)
        _save_matrices(outdir, 'svd-mtx-test.npy', svd_mtx_ts)
    else:
        raw_mtx_ts = raw_mtx
        svd_mtx_ts = svd_mtx
    # print('Applying ICA')
    # ica = FastICA(n_components=n_dim, max_iter=5, tol=0.01, whiten=True)
    # ica_mtx = ica.fit_transform(raw_mtx)
    # _save(outdir, 'ica-mxt.npy', ica_mtx)

    print('Applying DEEP PATIENT')
    dp_mtx = _deep_patient(raw_mtx, raw_mtx_ts, n_dim)
    _save_matrices(outdir, 'dp-mtx.npy', dp_mtx)
    if test_set:
        _save_matrices(outdir, 'dp-mtx-test.npy', dp_mtx)


"""
Private Functions
"""


# load data
def _load_test_dataset(indir):
    
    # read the vocabulary
    with open(path.join(indir, 'cohort-new-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        vcb = {r[1]: r[0] for r in rd}
 
    with open(path.join(indir, 'cohort_test-new-ehrseq.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        ehrs = {}
        for r in rd:
            ehrs.setdefault(r[0], list()).extend(list(map(int, r[1::])))
    print('Loaded test dataset with {0} patients'.format(
        len(ehrs)))

    # create raw data (scaled) counts
    mrns = [m for m in ehrs.keys()]
    data = ehrs.values()
    raw_dt = np.zeros((len(data), len(vcb)))
    for idx, token_list in enumerate(data):
        for t in token_list:
            raw_dt[idx, t - 1] += 1

    return (mrns, raw_dt, ehrs, vcb)


def _load_ehr_dataset(indir):
    # read the vocabulary
    with open(path.join(indir, 'cohort-new-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        vcb = {r[1]: r[0] for r in rd}

    # read raw data
    with open(path.join(indir, 'cohort-new-ehrseq.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        ehrs = {}
        for r in rd:
            ehrs.setdefault(r[0], list()).extend(list(map(int, r[1::])))
    print('Loaded dataset with {0} patients and {1} concepts'.format(
        len(ehrs), len(vcb)))

    # create raw data (scaled) counts
    mrns = [m for m in ehrs.keys()]
    data = ehrs.values()
    raw_dt = np.zeros((len(data), len(vcb)))
    for idx, token_list in enumerate(data):
        for t in token_list:
            raw_dt[idx, t - 1] += 1

    return (mrns, raw_dt, ehrs, vcb)


# run deep patient model

def _deep_patient(data_tr, data_ts, n_dim):
    sda = dp.SDA(data_tr.shape[1], nhidden=n_dim, nlayer=3,
                 param={'epochs': 10,
                        'batch_size': 4,
                        'corrupt_lvl': 0.05})
    sda.train(data_tr)
    return sda.apply(data_ts)


# save data

def _save_matrices(datadir, filename, data):
    outfile = path.join(datadir, filename)
    np.save(outfile, data)


def _save_mrns(datadir, data, filename):
    with open(path.join(datadir, filename), 'w') as f:
        f.write('\n'.join(data))


"""
Main Function
"""
def _process_args():
    parser = argparse.ArgumentParser(
             description='Baselines')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument('--test_set', dest='test_set', default=False,
                        help='Add fold')
    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__':
    args = _process_args()
    print ('')
    
    outdir = os.path.join(args.indir, 
                          '-'.join(['baselines',
                                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')]))

    start = time()
    run_baselines(indir=args.indir,
                  outdir = outdir,
                  test_set=args.test_set)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print ('Task completed\n')
