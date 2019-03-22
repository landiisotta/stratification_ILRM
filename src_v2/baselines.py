from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
import deep_patient as dp
from time import time
from os import path
import numpy as np
import random
import csv


def clustering_inspection(indir, outdir, disease_dt, n_dim=100,
                          sampling=-1, exclude_oth=True):

    print('Loading datasets')
    mrns, raw_data, vocab = _load_ehr_dataset(indir, sampling)

    print('\nRescaling the COUNT matrix')
    scaler = MinMaxScaler()
    raw_mtx = scaler.fit_transform(raw_data)

    print('Applying SVD')
    svd = TruncatedSVD(n_components=n_dim)
    svd_mtx = svd.fit_transform(raw_mtx)

    print('Applying ICA')
    ica = FastICA(n_components=n_dim, max_iter=5, tol=0.01, whiten=True)
    ica_mtx = ica.fit_transform(raw_mtx)

    print('Applying DEEP PATIENT')
    dp_mtx = _deep_patient(raw_mtx, n_dim)

    print('\nLoading ground truth data')
    gt_disease, disease_class = _load_ground_truth(indir, mrns)

    # remove disease with OTH condition if flag is enabled
    if exclude_oth:
        print('-- excluding OTH patients')
        idx = [i for i, m in enumerate(mrns) if gt_disease[m] != 'OTH']
        mrns = list(np.array(mrns)[idx])
        disease_class = list(np.array(disease_class)[idx])
        raw_mtx = raw_mtx[idx, :]
        svd_mtx = svd_mtx[idx, :]
        ica_mtx = ica_mtx[idx, :]
        dp_mtx = dp_mtx[idx, :]

    # print statistics
    disease_cnt = {}
    for d in disease_class:
        disease_cnt.setdefault(d, 1)
        disease_cnt[d] += 1
    print('-- no. of subjects: {0}'.format(len(gt_disease)))
    print('-- disease numerosities:')
    for k, v in disease_cnt.items():
        print('----- {0}: {1}'.format(k, v))

    # define clustering and plotting parameters
    HCpar = {'linkage_clu': 'ward',
             'affinity_clu': 'euclidean',
             'min_cl': 3,
             'max_cl': 11}

    print('\nEvaluating COUNT matrix')
    outer_clustering_analysis(raw_mtx,
                              disease_class,
                              HCpar['linkage_clu'],
                              HCpar['affinity_clu'])

    print('\nEvaluating SVD matrix')
    outer_clustering_analysis(svd_mtx,
                              disease_class,
                              HCpar['linkage_clu'],
                              HCpar['affinity_clu'])

    print('\nEvaluating ICA matrix')
    outer_clustering_analysis(ica_mtx,
                              disease_class,
                              HCpar['linkage_clu'],
                              HCpar['affinity_clu'])

    print('\nEvaluating DP matrix')
    outer_clustering_analysis(dp_mtx,
                              disease_class,
                              HCpar['linkage_clu'],
                              HCpar['affinity_clu'])


# external clustering analysis
def outer_clustering_analysis(data,
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
        print(
            '-- Cluster {0} -- '
            'Entropy: {1:.3f}, '
            'Purity: {2:.3f}, '
            'Max(p): {3}'.format(j, en, max_pu, ' & '.join(ds_max)))
        cweight = cnt_clu[j] / len(gt)
        entropy += cweight * en
        purity += cweight * max_pu

    print('Average Entropy: {0:.2f}'.format(entropy))
    print('Average Purity: {0:.2f}'.format(purity))
    print('Cluster Numerosity:')
    for c in set(clusters):
        print('-- cluster {0}: {1}'.format(c, clusters.count(c)))

    return clusters


# private functions

def _load_ehr_dataset(indir, sampling):
    # read the vocabulary
    with open(path.join(indir, 'cohort-new_vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        vcb = {r[1]: r[0] for r in rd}

    # read raw data
    with open(path.join(indir, 'cohort-new_ehr.csv')) as f:
        rd = csv.reader(f)
        ehrs = {}
        for r in rd:
            ehrs.setdefault(r[0], list()).extend(list(map(int, r[1::])))
    print('Loaded dataset with {0} patients and {1} concepts'.format(
        len(ehrs), len(vcb)))

    # sub-sample
    if sampling > 0:
        sa_mrns = [m for m in ehrs.keys()]
        random.shuffle(sa_mrns)
        sa_mrns = set(sa_mrns[:sampling])
        sa_ehrs = {k: v for k, v in ehrs.items() if k in sa_mrns}
        ehrs = sa_ehrs
        print('-- retained {0} patients'.format(len(ehrs)))

    # create raw data (scaled) counts
    mrns = [m for m in ehrs.keys()]
    data = ehrs.values()
    raw_dt = np.zeros((len(data), len(vcb)))
    for idx, token_list in enumerate(data):
        for t in token_list:
            raw_dt[idx, t - 1] += 1

    return (mrns, raw_dt, vcb)


def _load_ground_truth(indir, mrns):
    mrn_set = set(mrns)

    # list of diagnosed diseases associated with MRNs
    with open(path.join(indir, 'cohort-mrn_diseases.csv')) as f:
        rd = csv.reader(f)
        mrn_disease = {r[0]: r[1::] for r in rd}

    # ground truth using the first diagnosis
    gt_disease = {m: mrn_disease[m][0]
                  for m in mrn_disease if m in mrn_set}

    # define disease class
    disease_class = [gt_disease[m] for m in mrns]

    return (gt_disease, disease_class)


def _deep_patient(data, n_dim):
    sda = dp.SDA(data.shape[1], nhidden=n_dim, nlayer=3,
                 param={'epochs': 15,
                        'batch_size': 4,
                        'corrupt_lvl': 0.05})
    sda.train(data)
    return sda.apply(data)


# main function

if __name__ == '__main__':
    print ('')

    # define parameters
    datadir = '../data'
    dt_name = 'mixed'

    indir = path.join(datadir, dt_name)
    outdir = path.join(datadir, 'experiments',
                       '{0}-baselines'.format(dt_name))
    sampling = -1
    exclude_oth = True

    start = time()
    clustering_inspection(indir=indir,
                          outdir=outdir,
                          disease_dt=dt_name,
                          sampling=sampling,
                          exclude_oth=exclude_oth)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print ('Task completed\n')
