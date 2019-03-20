#!/usr/bin/env python
# coding: utf-8

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

##global variables
FRpar = {'n_terms':5,
         'ty_terms':['icd9', 'medication']}

# analyze clustering using silhouette scores
def silhouette_analysis(data,
                        min_clu,
                        max_clu,
                        linkage,
                        affinity,
                        preproc=False):
    if preproc:
        data = preprocessing.scale(data)
    # bound analysis range
    if min_clu < 2:
        min_clu = 2

    # run analysis for every clustering size
    best_silh = 0
    silh_scores = []
    for n in range(min_clu, max_clu, 1):
        hclu = AgglomerativeClustering(n_clusters=n,
                                       linkage=linkage,
                                       affinity=affinity)
        lbl = hclu.fit_predict(data).tolist()
        silh = silhouette_score(data, lbl, metric=affinity)
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
                                       linkage=linkage,
                                       affinity=affinity)
        n_clu = min_clu
        label = hclu.fit_predict(data).tolist()
        print('No. of Clusters: {0} -- Silhouette Score: {1:.3f}\n'.format(
            n_clu, best_silh))

    return (n_clu, label, silh_scores)


# SVD matrix of the TFIDF matrix of the raw ehr data
def svd_count(data, len_vocab, n_dimensions=200, tfidf=False):
    if tfidf:
    # apply tf-idf
        tfidf = TfidfTransformer()
        mtx = tfidf.fit_transform(data)
    else:
        scaler = MinMaxScaler()
        mtx = scaler.fit_transform(data)
    # reduce size of the matrix
    svd = TruncatedSVD(n_components=n_dimensions)
    svd_mtx = svd.fit_transform(mtx)

    return svd_mtx


# one plot with all the clusters
def single_plot(data, mrn_disease, colors, name):
    plt.figure(figsize=(20,10))
    for cl in set(mrn_disease):
        x = [d[0] for j, d in enumerate(data) if mrn_disease[j] == cl]
        y = [d[1] for j, d in enumerate(data) if mrn_disease[j] == cl]
        cols = [c for j, c in enumerate(colors) if mrn_disease[j] == cl]
        plt.xticks([])
        plt.yticks([])
        plt.scatter(x,y,c=cols, label=cl)
    plt.legend(loc=1)
    plt.savefig(name) 
    
# non-overlapping plots, one per cluster
def nonoverlap_plot(data, mrn_disease, colors):
    fig, ax = plt.subplots(len(set(mrn_disease)), 1, figsize=(20, 10*len(set(mrn_disease))))
    for idx, cl in enumerate(set(mrn_disease)):
        x = [d[0] for j, d in enumerate(data) if mrn_disease[j] == cl]
        y = [d[1] for j, d in enumerate(data) if mrn_disease[j] == cl]
        cols = [c for j, c in enumerate(colors) if mrn_disease[j] == cl]
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        ax[idx].scatter(x, y, c=cols, label=cl)
        ax[idx].legend()
        
# external clustering analysis
def outer_clustering_analysis(data, gt_clu, 
                              linkage, 
                              affinity, 
                              preproc=False):
    
    if preproc:
        data = preprocessing.scale(data)
        
    label_clu = sorted(set(gt_clu))

    # format clustering ground truth
    didx = {d: i for i, d in enumerate(label_clu)}
    idxd = {i:d for d, i in didx.items()}
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
            'Cluster: {0} -- '
            'Entropy: {1:.3f}, '
            'Purity: {2:.3f}'.format(j, en, max_pu))
        for d in ds_max:
            print("max(P) in cluster disease {0}".format(d))
        cweight = cnt_clu[j] / len(gt)
        entropy += cweight * en
        purity += cweight * max_pu

    print('Average Entropy: {0:.2f}'.format(entropy))
    print('Average Purity: {0:.2f}'.format(purity))
    
    return clusters

#Input: ehr lists corresponding to a cluster 
#Output: dictionary of term counts
def FreqDict(tokens):
    freq_dict = {}
    tok = []
    for seq in tokens:
        tok.extend(seq)
    tok = set(tok)
    for t in tok:
        for seq in tokens:
            if t in seq:
                if t not in freq_dict:
                    freq_dict[t] = 1
                else:
                    freq_dict[t] += 1
    return freq_dict
#Input: dictionary cluster:ehrs; list mrns
#Output:
def freq_term(data, pred_class, vocab, raw_ehr):
    list_terms = []
    for subc in range(len(set(pred_class))):
        tmp_data = {}
        for j in range(len(pred_class)):
            if pred_class[j] == subc:
                tmp_data.setdefault(subc, list()).append([rd for rd in data[j] 
                                                           if rd!=0 and 
                                                           (str.split(vocab[str(rd)], "::")[0] 
                                                           in FRpar['ty_terms'])])
        print("Cluster {0} numerosity: {1}".format(subc, len(tmp_data[subc])))
        term_count = FreqDict(tmp_data[subc])
        clust_mostfreq = []
        for l in range(FRpar['n_terms']):
            try:
                MFMT = max(term_count, key=(lambda key: term_count[key]))
                num_MFMT = 0
                subc_termc = 0
                for ehr in tmp_data[subc]:
                    for e in ehr:
                        if e == MFMT:
                            subc_termc += 1
                for seq in raw_ehr.values():
                    for t in seq:
                        if t == MFMT:
                            num_MFMT += 1
                print("% most frequent term:{0} "
                       "= {1:.2f} ({2} out of {3} terms in the whole dataset"
                       "-- N patients in cluster {4})".format(vocab[str(MFMT)], 
                                                              subc_termc/num_MFMT, 
                                                              subc_termc,
                                                              num_MFMT,
                                                              term_count[MFMT]))
                term_count.pop(MFMT)
                clust_mostfreq.append(MFMT)
            except ValueError:
                pass
        print("\n")
        list_terms.append(clust_mostfreq)
    return list_terms

##Hierarchical clustering function. Max silhouette.
def hclust_ehr(data, min_cl, max_cl, linkage, affinity):
    best_silh = -1
    list_silh = []
    for nc in range(min_cl,max_cl,1):
        hclust = AgglomerativeClustering(n_clusters=nc, 
                                         linkage=linkage, 
                                         affinity=affinity)
        tmp_label = hclust.fit_predict(data).tolist()
        tmp_silh = silhouette_score(data, tmp_label, metric=affinity)
        print(nc, tmp_silh)
        list_silh.append(float(tmp_silh))
        if tmp_silh > best_silh:
            best_silh = tmp_silh
            n_clust = nc
            label = tmp_label
    try:
        print("Number of clusters found:{0}, Silhouette score:{1:.3f}\n".format(n_clust, best_silh))
    except UnboundLocalError:
        hclust = AgglomerativeClustering(n_clusters=min_cl,
                                         linkage=linkage,
                                         affinity=affinity)
        n_clust = min_cl
        label = hclust.fit_predict(data).tolist()
        best_silh = silhouette_score(data, label)
        print("Number of clusters found:{0}, Silhouette score:{1:.3f}\n".format(n_clust, best_silh))
    return n_clust, label, list_silh

def chi_test(data, new_classes, term, mrns):
    count_mat = np.zeros((2, len(set(new_classes))), dtype=int)
    for c in set(new_classes):
        for idx, m in enumerate(mrns):
            if new_classes[idx] == c:
                if term in data[idx]:
                    count_mat[1][c] += 1
                else:
                    count_mat[0][c] += 1
    print("Count matrix:\n {0}".format(count_mat))
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(count_mat)
    string = "Chi-squared test statistics: chi2_stat = {0} -- p_val = {1} -- dof = {2}".format(
                                                                  chi2_stat,
                                                                  p_val,
                                                                  dof)#row = classes, columns = vocab
    print(string)
    
##Internal clustering validation
def inner_clustering_analysis(disease_class, data, raw_ehr, mrns, viz_data,
                              min_cl, max_cl, linkage, affinity, 
                              vocab, preproc=False):
    if preproc:
        data = preprocessing.scale(data)
    dis_viz_data = []
    subclass_dis = []
    for dis in sorted(set(disease_class)):
        tmp_data = []
        tmp_mrn = []
        tmp_raw_ehr = []
        for idx, d in enumerate(disease_class):
            if d == dis:
                dis_viz_data.append(viz_data[idx])
                tmp_data.append(data[idx])
                tmp_mrn.append(mrns[idx])
                tmp_raw_ehr.append(raw_ehr[mrns[idx]])
        print("Inspecting disease: {0}\n".format(dis))
        n_clust, label, _ = hclust_ehr(tmp_data, min_cl, max_cl, linkage, affinity)
        subclass_dis.extend([dis + ': subclust ' + str(l) for l in label])
        list_terms = freq_term(tmp_raw_ehr, label, vocab, raw_ehr)
        for l in range(len(set(label))):
            for lt in range(len(list_terms[l])):
                print("Odds ratio chi2 test for cluster {0}"
                      "term: {1}".format(l, vocab[str(list_terms[l][lt])]))
                try:
                    chi_test(tmp_raw_ehr, label, list_terms[l][lt], tmp_mrn)
                except ValueError:
                    print("empty class(es)")
                    pass
            print("\n\n")
    return(dis_viz_data, subclass_dis)


def clustering_inspection(indir,
                          expdir, 
                          disease_dt,
                          sampling=None,
                          exclude_oth=True):
    
    print("Loading datasets...")
    # get the list of diagnosed diseases associated with mrns
    with open(path.join(indir, 'cohort-mrn_diseases.csv')) as f:
        rd = csv.reader(f)
        mrn_disease = {r[0]: r[1::] for r in rd}
        
    # read encoded vectors file and ordered medical record numbers
    with open(path.join(expdir, 'mrns.csv')) as f:
        rd = csv.reader(f)
        mrns = [r[0] for r in rd]
    
    with open(path.join(expdir, 'encoded_vect.csv')) as f:
        rd = csv.reader(f)
        encoded = [list(map(float, r)) for r in rd]
        
        # sub-sample the collection
    if sampling is not None:
        idx = [i for i in range(len(mrns))]
        random.shuffle(idx)
        idx = idx[:n_samples]
        mrn_tmp = [mrns[i] for i in idx]
        enc_tmp = [encoded[i] for i in idx]
        mrns = mrn_tmp
        encoded = enc_tmp
    set_mrns = mrns

    # (1) first diagnosis
    gt_disease = {}
    if exclude_oth:
        for m in mrn_disease:
            if mrn_disease[m][0]!='OTH' and m in set_mrns:
                gt_disease[m] = mrn_disease[m][0]
            else:
                pass
    else:
        for m in mrn_disease:
            if m in set_mrns:
                gt_disease[m] = mrn_disease[m][0]
            else:
                pass
                    
    # read the vocabulary
    with open(path.join(indir, 'cohort-new_vocab.csv')) as f:    
        rd = csv.reader(f)
        next(rd)
        vocab = {r[1]: r[0] for r in rd}
    len_vocab = len(vocab)

    # read raw data
    with open(path.join(indir, 'cohort-new_ehr.csv')) as f:
        rd = csv.reader(f)
        raw_ehr = {}
        for r in rd:
            if r[0] in gt_disease.keys():
                raw_ehr.setdefault(r[0], list()).extend(list(map(int, r[1::])))
                
    ##Read LSTM encoded vectors file and ordered medical record numbers
#    with open(expdir + '/LSTMencoded_vect.csv') as f:
#        rd = csv.reader(f)
#        lstm_encoded_vect = []
#        for r in rd:
#            lstm_encoded_vect.append(list(map(float, r)))
        
#    with open(expdir + '/LSTMmrns.csv') as f:
#        rd = csv.reader(f)
#        lstm_mrns = [r[0] for r in rd]

    tmp_mrns = []
 #   tmp_lstm_mrns = []
    tmp_encoded = []
 #   tmp_lstm_encoded = []
 #   for (idx, m), m_lstm in zip(enumerate(set_mrns), lstm_mrns):
    for idx, m in enumerate(set_mrns): 
        if m in gt_disease.keys():
            tmp_mrns.append(m)
            tmp_encoded.append(encoded[idx])
            #elif m_lstm in gt_disease.keys():
            #    tmp_lstm_mrns.append(m_lstm)
            #    tmp_lstm_encoded.append(lstm_encoded_vect[idx])
        else:
            pass
    set_mrns = tmp_mrns
#    lstm_mrns = tmp_lstm_mrns
    encoded = tmp_encoded
#    lstm_encoded_vect = tmp_lstm_encoded

    # raw data (scaled) counts
    scaler = MinMaxScaler()
    data = raw_ehr.values()
    mrn_list = [m for m in raw_ehr.keys()]
    raw_data = np.zeros((len(data), len_vocab))
    for idx, token_list in enumerate(data):
        for t in token_list:
            raw_data[idx, t - 1] += 1

    disease_count = {}
    for d in gt_disease.values():
        if d not in disease_count:
            disease_count[d] = 1
        else:
            disease_count[d] += 1
    print("Number of subjects:{0}".format(len(gt_disease)))
    print("Disease numerosities:\n {0}".format(disease_count))
    
    # plot colors
    col_dict = matplotlib.colors.CSS4_COLORS
    c_out = ['mintcream', 'cornsilk', 'lavenderblush', 'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'powderblue', 'floralwhite', 'ghostwhite',
     'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
     'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'linen', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
     'peachpuff', 'mistyrose', 'lemonchiffon', 'lightblue', 'seashell', 'white', 'blanchedalmond', 'oldlace', 'moccasin', 'snow', 'darkgray',
     'ivory', 'whitesmoke']
    colormap = [c for c in col_dict if c not in c_out]

    # choose the disease classes: first_disease, oth_disease
    disease_class_first = [gt_disease[m] for m in set_mrns]
    raw_disease_class_first = [gt_disease[m] for m in mrn_list]
#    lstm_disease_class_first = [gt_disease[m] for m in lstm_mrns]
    disease_dict = {d: i for i, d in enumerate(set(disease_class_first))}
    
    ##Parameters for CNN-AE and LSTM
    HCpar = {'linkage_clu':'ward',
             'affinity_clu':'euclidean',
             'min_cl':3,
             'max_cl':11}

    print("UMAP embeddings for CNN-AE encodings...")
    # initialize UMAP
    reducer = umap.UMAP(n_neighbors=200, min_dist=0.5, metric = HCpar['affinity_clu'], n_components=2)

    # UMAP on the CNN encoded vectors
    encoded_umap = reducer.fit_transform(encoded).tolist()
    print('Computed: CNN - AE encoded vectors umap') 
    
    print("Evaluating CNN-AE encodings...")
    ##CNN-AE encodings
    # plot data
    colors_en1 = [colormap[disease_dict[v]] for v in disease_class_first]
    single_plot(encoded_umap, disease_class_first, colors_en1, 
                path.join(expdir, 'cnn-ae_encodings_plot.png'))
    # plot cluster results
    clusters = outer_clustering_analysis(encoded, disease_class_first, HCpar['linkage_clu'], HCpar['affinity_clu'], preproc=False)
    for c in set(clusters):
        print("Cluster {0} numerosity = {1}".format(c, clusters.count(c)))

    colors_en2 = [colormap[v] for v in clusters]
    single_plot(encoded_umap, clusters, colors_en2, 
                path.join(expdir, 'cnn-ae_outer-clust_plot.png'))
    # inner clustering analysis
    encoded_subplots, en_sub_clust = inner_clustering_analysis(disease_class_first, encoded, raw_ehr,
                                                               set_mrns, encoded_umap, 
                                                               HCpar['min_cl'], HCpar['max_cl'],
                                                               HCpar['linkage_clu'], HCpar['affinity_clu'], 
                                                               vocab, preproc=False)
    encoded_new_disease_dict = {}
    for idx, nd in enumerate(set(en_sub_clust)):
        encoded_new_disease_dict[nd] = idx
    colors_en3 = [colormap[encoded_new_disease_dict[v]] for v in en_sub_clust]
    single_plot(encoded_subplots, en_sub_clust, colors_en3, 
                path.join(expdir, 'cnn-ae_sub-clust_plot.png'))
 
    # UMAP on the LSTM encoded vectors
 #   print("UMAP embedding for LSTM encodings...")
 #   lstm_encoded_umap = reducer.fit_transform(lstm_encoded_vect).tolist()
 #   print("Computed: LSTM encoded vectors umap")

 #   print("Evaluating LSTM encodings...")
    ##LSTM encodings
    # plot data
 #   colors_lstm1 = [colormap[disease_dict[v]] for v in lstm_disease_class_first]
 #   single_plot(lstm_encoded_umap, lstm_disease_class_first, colors_lstm1, 
 #               path.join(expdir, 'lstm_encodings_plot.png'))
    # plot cluster results
 #   clusters = outer_clustering_analysis(lstm_encoded_vect, lstm_disease_class_first, HCpar['linkage_clu'], 
 #                                        HCpar['affinity_clu'], preproc=False)
 #   for c in set(clusters):
 #       print("Cluster {0} numerosity = {1}".format(c, clusters.count(c)))
    
 #   colors_lstm2 = [colormap[v] for v in clusters]
 #   single_plot(lstm_encoded_umap, clusters, colors_lstm2, 
 #               path.join(expdir, 'lstm_outer-clust_plot.png'))
 #   lstm_subplots, lstm_sub_clust = inner_clustering_analysis(lstm_disease_class_first, lstm_encoded_vect, 
 #                                                             raw_ehr, lstm_mrns, lstm_encoded_umap,
 #                                                             HCpar['min_cl'], HCpar['max_cl'],
 #                                                             HCpar['linkage_clu'], HCpar['affinity_clu'],
 #                                                             vocab, preproc=False)
 #   lstm_new_disease_dict = {}
 #   for idx, nd in enumerate(set(lstm_sub_clust)):
 #       lstm_new_disease_dict[nd] = idx
 #   colors_lstm3 = [colormap[lstm_new_disease_dict[v]] for v in lstm_sub_clust]
 #   single_plot(lstm_subplots, lstm_sub_clust, colors_lstm3, 
 #               path.join(expdir, 'lstm_sub-clust_plot.png'))
    
    ##Silhouette analysis
    _,_,enc_silh = silhouette_analysis(encoded, HCpar['min_cl'], HCpar['max_cl'],
                                       HCpar['linkage_clu'], HCpar['affinity_clu'])
 #   _,_,lstm_silh = silhouette_analysis(lstm_encoded_vect, HCpar['min_cl'], HCpar['max_cl'],
 #                                       HCpar['linkage_clu'], HCpar['affinity_clu'])
    
    ##Parameters for COUNT baseline
    HCpar = {'linkage_clu':'ward',
             'affinity_clu':'euclidean',
             'min_cl':3,
             'max_cl':11}
    
    print("UMAP embedding for COUNT-SVD matrix...")
    # initialize UMAP
    reducer = umap.UMAP(n_neighbors=200, min_dist=0.5, metric = HCpar['affinity_clu'], n_components=2)
    # UMAP on the TF-IDF + SVD matrix
    svd_mat = svd_count(raw_data, len_vocab, n_dimensions = 100)
    count_umap = reducer.fit_transform(svd_mat).tolist()
    print("Computed: COUNT matrix umap")

    print("Evaluating scaled COUNT matrix...")
    ##COUNT scaled matrix
    # plot data
    colors_count1 = [colormap[disease_dict[v]] for v in raw_disease_class_first]
    single_plot(count_umap, raw_disease_class_first, colors_count1, 
                path.join(expdir, 'count_encodings_plot.png'))
    # plot cluster results
    clusters = outer_clustering_analysis(svd_mat, raw_disease_class_first, HCpar['linkage_clu'], 
                                         HCpar['affinity_clu'], preproc=False)
    for c in set(clusters):
        print("Cluster {0} numerosity = {1}".format(c, clusters.count(c)))

    colors_count2 = [colormap[v] for v in clusters]
    single_plot(count_umap, clusters, colors_count2, 
                path.join(expdir, 'count_outer-clust_plot.png'))
    count_subplots, count_sub_clust = inner_clustering_analysis(raw_disease_class_first, svd_mat, 
                                                                raw_ehr, mrn_list, count_umap,
                                                                HCpar['min_cl'], HCpar['max_cl'],
                                                                HCpar['linkage_clu'], HCpar['affinity_clu'],
                                                                vocab, preproc=False)
    count_new_disease_dict = {}
    for idx, nd in enumerate(set(count_sub_clust)):
        count_new_disease_dict[nd] = idx
    colors_count3 = [colormap[count_new_disease_dict[v]] for v in count_sub_clust]
    single_plot(count_subplots, count_sub_clust, colors_count3, 
                path.join(expdir, 'count_sub-clust_plot.png'))
    
    ##Silhouette analysis
    _,_,count_silh = silhouette_analysis(svd_mat,HCpar['min_cl'], HCpar['max_cl'],
                                         HCpar['linkage_clu'], HCpar['affinity_clu'])
    
    print("Plotting silhouettes for model and baselines:")
    minmax_clust = [r for r in range(HCpar['min_cl'], HCpar['max_cl'])]
    fig, ax = plt.subplots()
    ax.plot(minmax_clust, enc_silh, '-o')
 #   ax.plot(minmax_clust, lstm_silh, '-o')
    ax.plot(minmax_clust, count_silh, '-o')
 #   ax.legend(['cnn-ae', 'lstm', 'count matrix'])
    ax.legend(['cnn-ae', 'count matrix'])
    fig.savefig(path.join(expdir, 'silhouette_plot.png'))

def _process_args():
    parser = argparse.ArgumentParser(
        description='EHR Patient Stratification: perform hierarchical clustering '
        'validate results and inspect subclusters.')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument(dest='expdir', help='Experiment directory')
    parser.add_argument(dest='disease_dt', help='Disease dataset name')
    parser.add_argument('--oth', default=True, type=bool,
                       help='Drop OTH for clustering and visualization' 
                       '(default:True)')
    parser.add_argument('-s', default=None, type=int,
                        help='Enable sub-sampling with data size '
                        '(default: None)')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print ('')

    start = time()
    clustering_inspection(indir=args.indir,
                          expdir=args.expdir,
                          disease_dt=args.disease_dt,
                          sampling=args.s,
                          exclude_oth=args.oth)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print ('Task completed\n')

