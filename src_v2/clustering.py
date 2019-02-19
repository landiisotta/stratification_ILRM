from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import csv


# hierarchical clustering
def hclust_ehr(data, min_cl, max_cl, metric, full_log=True):
    best_silh = 0.0
    list_silh = []
    for n in range(min_cl, max_cl, 1):
        hclust = AgglomerativeClustering(n_clusters=n,
                                         linkage='ward',
                                         affinity=metric)
        tmp_label = hclust.fit_predict(data).tolist()
        tmp_silh = silhouette_score(data, tmp_label)
        if tmp_silh < 0:
            return
        if full_log:
            print(' -- {0}: {1:.3f}'.format(n, tmp_silh))
        list_silh.append(float(tmp_silh))
        if tmp_silh > best_silh:
            best_silh = tmp_silh
            n_clust = n
            label = tmp_label
    try:
        print('No. of clusters: {0} -- Silhouette Score: {1:.3f}'.format(
            n_clust, best_silh))

    except UnboundLocalError:
        hclust = AgglomerativeClustering(n_clusters=min_cl,
                                         linkage='complete',
                                         affinity=metric)
        n_clust = min_cl
        label = hclust.fit_predict(data).tolist()
        print('No. of Clusters: {0} -- Silhouette Score: {1:.3f}'.format(
            n_clust, best_silh))

    return n_clust, label, list_silh


# SVD matrix of the TFIDF matrix of the raw ehr data
def svd_tfidf(datafile, len_vocab, n_dimensions=64):
    data = _load_raw_data(datafile)

    # format data
    count_mtx = np.zeros((len(data), len_vocab))
    for idx, token_list in enumerate(data):
        for t in token_list:
            if t != 0:
                count_mtx[idx, t - 1] += 1

    # apply tf-idf
    tfidf = TfidfTransformer()
    tfidf_mtx = tfidf.fit_transform(count_mtx)

    # reduce size of the matrix
    svd = TruncatedSVD(n_components=n_dimensions)
    svd_mtx = svd.fit_transform(tfidf_mtx)

    return svd_mtx


"""
private functions
"""


def _load_raw_data(filename):
    with open(filename) as f:
        rd = csv.reader(f)
        raw_ehr = {}
        for r in rd:
            raw_ehr.setdefault(r[0], list()).extend(list(map(int, r[1::])))
    return raw_ehr.values()
