from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import numpy as np
import csv


# run the full clustering evaluation
def eval_hierarchical_clustering(data,
                                 gt_clu,
                                 min_clu,
                                 max_clu,
                                 affinity_clu='euclidean',
                                 preproc=False):

    print len(data[0])

    # normalize data
    if preproc:
        data = preprocessing.scale(data)

    # silhuoette analysis
    silhouette_analysis(data, min_clu, max_clu, affinity_clu)

    # external cluster analysis
    outer_clustering_analysis(data, gt_clu, affinity_clu)

    return


# analyze clustering using silhouette scores
def silhouette_analysis(data,
                        min_clu=2,
                        max_clu=10,
                        affinity_clu='euclidean'):
    # bound analysis range
    if min_clu < 2:
        min_clu = 2

    # run analysis for every clustering size
    best_silh = 0
    silh_scores = []
    for n in range(min_clu, max_clu, 1):
        hclu = AgglomerativeClustering(n_clusters=n,
                                       linkage='ward',
                                       affinity=affinity_clu)
        lbl = hclu.fit_predict(data).tolist()
        silh = silhouette_score(data, lbl)
        if silh < 0:
            break
        print(' -- {0}: {1:.3f}'.format(n, silh))
        silh_scores.append(silh)
        if silh > best_silh:
            best_silh = silh
            n_clu = n
            label = lbl
    try:
        print('No. of clusters: {0} -- Silhouette Score: {1:.3f}\n'.format(
            n_clu, best_silh))

    except UnboundLocalError:
        hclu = AgglomerativeClustering(n_clusters=min_clu,
                                       linkage='complete',
                                       affinity=affinity_clu)
        n_clu = min_clu
        label = hclu.fit_predict(data).tolist()
        print('No. of Clusters: {0} -- Silhouette Score: {1:.3f}\n'.format(
            n_clu, best_silh))

    return (n_clu, label, silh_scores)


# external clustering analysis
def outer_clustering_analysis(data, gt_clu, affinity_clu):
    label_clu = sorted(set(gt_clu))

    # format clustering ground truth
    didx = {d: i for i, d in enumerate(label_clu)}
    gt = [didx[d] for d in gt_clu]

    # validate cluster number
    if len(label_clu) == 1:
        n_clu = 3
    else:
        n_clu = len(label_clu)

    # run clustering
    hclust = AgglomerativeClustering(n_clusters=n_clu,
                                     linkage='ward',
                                     affinity=affinity_clu)
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
        pu = max(pu)
        print(
            'Cluster: {0} -- '
            'Entropy: {1:.3f}, '
            'Purity: {2:.3f}'.format(j, en, pu))

        cweight = cnt_clu[j] / len(gt)
        entropy += cweight * en
        purity += cweight * pu

    print('Average Entropy: {0:.2f}'.format(entropy))
    print('Average Purity: {0:.2f}'.format(purity))


"""
Baselines
"""


# SVD matrix of the TFIDF matrix of the raw ehr data
def svd_tfidf(datafile, len_vocab, n_dimensions=200):
    data = load_raw_data(datafile)

    # format data
    count_mtx = np.zeros((len(data), len_vocab))
    for idx, token_list in enumerate(data):
        for t in token_list:
            count_mtx[idx, t - 1] += 1

    # apply tf-idf
    tfidf = TfidfTransformer()
    tfidf_mtx = tfidf.fit_transform(count_mtx)

    # reduce size of the matrix
    svd = TruncatedSVD(n_components=n_dimensions)
    svd_mtx = svd.fit_transform(tfidf_mtx)

    return svd_mtx


"""
Load data
"""


# load associations MRN - First Disease
def load_mrn_disease(filename, mrn):
    with open(filename) as f:
        rd = csv.reader(f)
        mrn_disease = {}
        for r in rd:
            mrn_disease.setdefault(r[0], list()).append(r[1])
    return [mrn_disease[m][0] for m in mrn]


# load roaw data for baseline comparison
def load_raw_data(filename):
    with open(filename) as f:
        rd = csv.reader(f)
        raw_ehr = {}
        for r in rd:
            raw_ehr.setdefault(r[0], list()).extend(list(map(int, r[1::])))
    return raw_ehr.values()
