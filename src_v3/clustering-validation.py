from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import numpy as np
import csv
import utils as ut
from utils import HCpar
import re
import argparse
import sys
from time import time
import os
import scipy.sparse

def clustering_inspection(datadir, indir, encdir, list_model, 
                          code, n_iter):
    with open(os.path.join(indir, 'cohort-new-vocab.csv')) as f:
        vocab = {}
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            vocab[r[0]] = r[1]

    print('Loading data...\n\n')
    gt_disease = _load_ground_truth(datadir,
                                    code, n_iter)
    par_dict = {}
    for mod in list_model:
        if mod == 'convae':
            with open(os.path.join(encdir, mod + '-avg_vect.csv')) as f:
                rd = csv.reader(f)
                next(rd)
                mrn_list = [] 
                mtx = []
                for r in rd:
                    if r[0] in gt_disease[code].keys():
                        mtx.append(r[1::])
                        mrn_list.append(r[0])
        else:
            if mod == 'raw':
                mtx = scipy.sparse.load_npz(os.path.join(encdir, mod + '-mtx.npz'))
            else:
                mtx = np.load(os.path.join(encdir, mod + '-mtx.npy'))
                mtx = scipy.sparse.csr_matrix(mtx)
            with open(os.path.join(encdir, 'bs-mrn.txt')) as f:
                mrn_list = f.read().splitlines()
            idx_list = []
            mrn_list_tmp = []
            for idx, m in enumerate(mrn_list):
                if m in gt_disease[code].keys():
                    mrn_list_tmp.append(m)
                else:
                    idx_list.append(idx)
            mrn_list = mrn_list_tmp
            mtx = _delete_rows_csr(mtx, idx_list)
            mtx = mtx.todense()
        disease_class = {code: [gt_disease[code][m] for m in mrn_list]}
        print("Model {0} -- Disease code {1}".format(mod.upper(), code.upper()))
        print("Disease numerosities:")
        dis_a = np.array(disease_class[code])
        unique, counts = np.unique(dis_a, return_counts=True)
        for dis, n in dict(zip(unique, counts)).items():
            print("{0} -- {1}".format(dis, n))
        print('\n')
        ent_s, pur_s, n_disf = cluster_evaluation(encdir, mtx,
                                                  disease_class[code],
                                                  HCpar,
                                                  mod,
                                                  code, n_iter)
        par_dict[mod] = [ent_s, pur_s, n_disf]
        print('*'*100)
        print('\n')
    return par_dict


def cluster_evaluation(encdir, data_mtx, gt_clu,
                       HCpar, model_name,
                       code, n_iter):

    label = model_name.upper()
    print('Evaluating {0} embeddings\n'.format(label))
    
    # evaluate cluster results
    clu, ent, pur, n_disf = _outer_clustering_analysis(
                           data_mtx, gt_clu, 
                           HCpar['linkage_clu'], 
                           HCpar['affinity_clu'])

    with open(os.path.join(encdir, 'cl-subsampling/outer-cl-{0}-{1}-it{2}.txt'.format(model_name, 
                                                                 code, n_iter)), 'w') as f:
        wr = csv.writer(f)
        wr.writerows([[c] for c in clu])

    return ent, pur, n_disf

"""
Private Functions
"""


def _delete_rows_csr(mat, indices):
    """
    Remove the rows in list indices form the CSR sparse matrix
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        print("Matrix not in CSR format, applying .tocsr()\n")
        mat = mat.tocsr()
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def _outer_clustering_analysis(data,
                               gt_clu,
                               linkage,
                               affinity):

    label_clu = sorted(set(gt_clu))

    # format clustering ground truth
    didx = {d: i for i, d in enumerate(label_clu)}
    idxd = {i: d for d, i in didx.items()}
    gt = [didx[d] for d in gt_clu]

    # validate cluster number
    if len(label_clu) == 1:
        n_clu = 3
    else:
        n_clu = len(label_clu)

    data = np.array(data).astype(np.float32)
    # run clustering
    hclust = AgglomerativeClustering(n_clusters=n_clu,
                                     linkage=linkage,
                                     affinity=affinity)
    clusters = hclust.fit_predict(data).tolist()

    # count cluster occurrences
    cnt_clu = [0] * n_clu
    for c in clusters:
        cnt_clu[c] += 1
    class_clu = [[0] * n_clu for _ in range(len(label_clu))]
    for i, gi in enumerate(gt):
        class_clu[gi][clusters[i]] += 1

    # compute entropy and purity
    entropy = 0
    purity = 0
    dis_found = set()
    for j in range(0, max(clusters) + 1):
        en = 0
        pu = []
        for i in range(0, max(gt) + 1):
            pij = class_clu[i][j] / cnt_clu[j]
            pu.append(pij)
            if pij != 0:
                en += -(pij * np.log2(pij))
        max_pu = max(pu)
        ds_max = []
        for idx, p in enumerate(pu):
            if p == max_pu:
                ds_max.append(idxd[idx])
                dis_found.add(idxd[idx])
        print(
            '-- Cluster {0} -- '
            'Entropy: {1:.3f}, '
            'Purity: {2:.3f}, '
            'Max(p): {3}'.format(j, en, max_pu, ' & '.join(ds_max)))
        cweight = cnt_clu[j] / len(gt)
        entropy += cweight * en
        purity += cweight * max_pu

    print('\nAverage Entropy: {0:.2f}'.format(entropy))
    print('Average Purity: {0:.2f}'.format(purity))
    print('Number of diseases identified: %d' % len(dis_found))
    print('Cluster Numerosity:')
    for c in set(clusters):
        print('-- cluster {0}: {1}'.format(c, clusters.count(c)))

    return clusters, entropy, purity, len(dis_found)


def _load_ground_truth(datadir, code, n_iter):
    
    gt_disease = {code: {}}
    # list of diagnosed diseases associated with MRNs
    with open(os.path.join(datadir, 
              'snomed_subsampling/patient-5000-disease-{0}-it{1}.csv'.format(code, 
                                                                      n_iter))) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            gt_disease[code][r[0]] = r[1]

    return gt_disease


"""
Main Function
"""


def _process_args():
    parser = argparse.ArgumentParser(
          description='Outer clustering encodings evaluation')
    parser.add_argument(dest='datadir',
          help='Directory where disease are stored')
    parser.add_argument(dest='indir',
          help='Directory with the processed data')
    parser.add_argument(dest='encdir',
          help='Directory with the encodings'),
    parser.add_argument(dest='code',
          help='Specify if SNOMED or CCS-SINGLE')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print('')

    start = time()
    
    ent_dict = {}
    pur_dict = {}
    ndisf_dict = {} # number of diseases found
    for it in range(ut.n_iter):
        print("Cohort subsampling N {0}".format(it)) 
        par_dict = clustering_inspection(datadir=args.datadir,
                                             indir=args.indir,
                                             encdir=args.encdir,
                                             list_model=ut.ev_model,
                                             code=args.code,
                                             n_iter=it)
        for mod in par_dict:
            ent_dict.setdefault(mod, list()).append(par_dict[mod][0])
            pur_dict.setdefault(mod, list()).append(par_dict[mod][1])
            ndisf_dict.setdefault(mod, list()).append(par_dict[mod][2])
    # sorted estimates for bootstrapped CI
    for mod in ent_dict.keys():
        ent_dict[mod] = sorted(ent_dict[mod])
        pur_dict[mod] = sorted(pur_dict[mod])
        print("Model {0}".format(mod.upper()))
        print("Mean Entropy: {0:.3f} (SD = {1:.3f}) -- CI = [{2:.3f}, {3:.3f}]".format(np.mean(ent_dict[mod]), 
                                                                np.std(ent_dict[mod]),
                                                                ent_dict[mod][int(0.25 * ut.n_iter)],
                                                                ent_dict[mod][int(0.975 * ut.n_iter)]))
        print("Mean Purity: {0:.3f} (SD = {1:.3f}) -- CI = [{2:.3f}, {3:.3f}]".format(np.mean(pur_dict[mod]),
                                                                np.std(pur_dict[mod]),
                                                                pur_dict[mod][int(0.25 * ut.n_iter)],
                                                                pur_dict[mod][int(0.975 * ut.n_iter)]))
        print("Mean number of detected diseases: {0:.3f} (SD = {1:.3f})\n".format(np.mean(ndisf_dict[mod]),
                                                                np.std(ndisf_dict[mod])))
    print('\nProcessing time: %s seconds' % round(time()-start, 2))

    print('Task completed\n') 
