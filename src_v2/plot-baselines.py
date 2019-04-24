import os
import csv
import random
import umap
import numpy as np
from os import path
from scipy import stats
from sklearn import preprocessing
from sklearn.manifold.t_sne import TSNE
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
import argparse
import sys
from time import time

# one plot with all the clusters
def single_plot(data, mrn_disease, colors, name, leg_labels=None):
    plt.figure(figsize=(20,10))
    for cl in set(mrn_disease):
        x = [d[0] for j, d in enumerate(data) if mrn_disease[j] == cl]
        y = [d[1] for j, d in enumerate(data) if mrn_disease[j] == cl]
        cols = [c for j, c in enumerate(colors) if mrn_disease[j] == cl]
        plt.xticks([])
        plt.yticks([])
        plt.scatter(x,y,c=cols, label=cl)
    if leg_labels is not None:
        plt.legend(loc=3, labels=leg_labels, markerscale=2, fontsize=12)
    else:
        plt.legend(loc=3, markerscale=2, fontsize=12)
    plt.savefig(name, formap='eps', dpi=1000)

indir = os.path.join(os.path.expanduser('~/data1/stratification_ILRM/data/'), 'ehr100k')
expdir = os.path.join(os.path.expanduser('~/data1/stratification_ILRM/data/experiments/'), 'ehr100k-w2v-softplus')

with open(path.join(indir, 'cohort-mrn_diseases.csv')) as f:
    rd = csv.reader(f)
    mrn_disease = {r[0]: r[1::] for r in rd}

    # read encoded vectors file and ordered medical record numbers
with open(path.join(expdir, 'mrn-raw.txt')) as f:
    rd = csv.reader(f)
    mrn_raw = [r[0] for r in rd]
with open(path.join(expdir, 'mrn-svd.txt')) as f:
    rd = csv.reader(f)
    mrn_svd = [r[0] for r in rd]
with open(path.join(expdir, 'mrn-dp.txt')) as f:
    rd = csv.reader(f)
    mrn_dp = [r[0] for r in rd]

raw_mxt = np.load(path.join(expdir, 'raw-mxt.npy')).tolist()
svd_mxt = np.load(path.join(expdir, 'svd-mxt.npy')).tolist()
dp_mxt = np.load(path.join(expdir, 'dp-mxt.npy')).tolist()

gt_disease = {}
for m in mrn_disease:
    if mrn_disease[m][0]!='OTH' and m in mrn_raw:
        gt_disease[m] = mrn_disease[m][0]
    else:
        pass

raw_disease_class_first = [gt_disease[m] for m in mrn_raw if m in gt_disease.keys()]
svd_disease_class_first = [gt_disease[m] for m in mrn_svd if m in gt_disease.keys()]
dp_disease_class_first = [gt_disease[m] for m in mrn_dp if m in gt_disease.keys()]
disease_dict = {d: i for i, d in enumerate(['Multiple Myeloma', 'Prostate Cancer', 'Diabetes', 'Alzheimer\'s disease',
                                            'Parkinson\'s disease',
                                            'Breast Cancer'])}

raw_tmp = []
for idx, m in enumerate(mrn_raw):
    if m in gt_disease.keys():
        raw_tmp.append(raw_mxt[idx])
raw_mxt = raw_tmp

svd_tmp = []
for idx, m in enumerate(mrn_svd):
    if m in gt_disease.keys():
        svd_tmp.append(svd_mxt[idx])
svd_mxt = svd_tmp

dp_tmp = []
for idx, m in enumerate(mrn_dp):
    if m in gt_disease.keys():
        dp_tmp.append(dp_mxt[idx])
dp_mxt = dp_tmp

print("UMAP embeddings for CNN-AE encodings...")
    # initialize UMAP
reducer = umap.UMAP(n_neighbors=200, min_dist=0.5, metric = 'euclidean', n_components=2)

    # UMAP on the CNN encoded vectors
raw_umap = reducer.fit_transform(raw_mxt).tolist()
svd_umap = reducer.fit_transform(svd_mxt).tolist()
dp_umap = reducer.fit_transform(dp_mxt).tolist()

colors_vec = ['b','g','m','darkorange','dimgrey','saddlebrown']
colors_raw = [colors_vec[disease_dict[v]] for v in raw_disease_class_first]
single_plot(raw_umap, raw_disease_class_first, colors_raw,
            path.join(expdir, 'raw_baseline_plot.eps'))

colors_svd = [colors_vec[disease_dict[v]] for v in svd_disease_class_first]
single_plot(svd_umap, svd_disease_class_first, colors_svd,
            path.join(expdir, 'svd_baseline_plot.eps'))

colors_dp = [colors_vec[disease_dict[v]] for v in dp_disease_class_first]
single_plot(dp_umap, dp_disease_class_first, colors_dp,
            path.join(expdir, 'dp_baseline_plot.eps'))
