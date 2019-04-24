from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from scipy import stats
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from os import path
import umap
import matplotlib
import matplotlib.pyplot as plt
import deep_patient as dp
from time import time
import numpy as np
import random
import csv
import os


FRpar = {'n_terms': 5,
         'ty_terms': ['icd9', 'medication']}


def clustering_inspection(indir, outdir, disease_dt, n_dim=100,
                          sampling=-1, exclude_oth=True):

    # create output directory
    try:
        os.mkdir(outdir)
    except OSError:
        pass

    colormap = _define_colormap()

    print('Loading datasets')
    mrns, raw_data, ehrs, vocab = _load_ehr_dataset(indir, sampling)
    _save_mrns(outdir, mrns)

    print('\nRescaling the COUNT matrix')
    scaler = MinMaxScaler()
    raw_mtx = scaler.fit_transform(raw_data)
    _save_matrices(outdir, 'raw-mxt.npy', raw_mtx)

    print('Applying SVD')
    svd = TruncatedSVD(n_components=n_dim)
    svd_mtx = svd.fit_transform(raw_mtx)
    _save_matrices(outdir, 'svd-mxt.npy', svd_mtx)

    # print('Applying ICA')
    # ica = FastICA(n_components=n_dim, max_iter=5, tol=0.01, whiten=True)
    # ica_mtx = ica.fit_transform(raw_mtx)
    # _save(outdir, 'ica-mxt.npy', ica_mtx)

    print('Applying DEEP PATIENT')
    dp_mtx = _deep_patient(raw_mtx, n_dim)
    _save_matrices(outdir, 'dp-mxt.npy', dp_mtx)

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
        # ica_mtx = ica_mtx[idx, :]
        dp_mtx = dp_mtx[idx, :]

    _print_dataset_statistics(gt_disease, disease_class)

    # define clustering and plotting parameters
    HCpar = {'linkage_clu': 'ward',
             'affinity_clu': 'euclidean',
             'min_cl': 3,
             'max_cl': 11}

    # evaluate models
    cluster_evaluation(svd_mtx, disease_class, ehrs,
                       mrns, vocab, colormap, HCpar, 'svd')

    cluster_evaluation(dp_mtx, disease_class, ehrs,
                       mrns, vocab, colormap, HCpar, 'dp')

    cluster_evaluation(raw_mtx, disease_class, ehrs,
                       mrns, vocab, colormap, HCpar, 'raw')

    return


def cluster_evaluation(data_mtx, gt_clu, ehrs, mrns, vocab,
                       colormap, HCpar, model_name):

    label = model_name.upper()
    print('Evaluating {0} embeddings\n'.format(label))

    # initialize UMAP
    reducer = umap.UMAP(n_neighbors=200, min_dist=0.1,
                        metric=HCpar['affinity_clu'], n_components=2)

    # apply UMAP to the embeddings
    umap_mtx = reducer.fit_transform(data_mtx).tolist()
    print('Computed: {0} vectors umap'.format(label))

    disease_dict = {d: i for i, d in enumerate(set(gt_clu))}
    colors = [colormap[disease_dict[v]] for v in gt_clu]
    _single_plot(umap_mtx, gt_clu, colors,
                 path.join(outdir, '{0}-encodings.png'.format(model_name)))

    # evaluate cluster results
    clu = _outer_clustering_analysis(
        data_mtx, gt_clu, HCpar['linkage_clu'], HCpar['affinity_clu'])

    colors = [colormap[v] for v in clu]
    _single_plot(umap_mtx, clu, colors,
                 path.join(outdir, '{0}-outer-cluster.png'.format(model_name)))

    # inner clustering analysis
    subplots, sub_clust = _inner_clustering_analysis(
        gt_clu, data_mtx, ehrs, mrns,
        umap_mtx,
        HCpar['min_cl'], HCpar['max_cl'],
        HCpar['linkage_clu'], HCpar['affinity_clu'], vocab)

    # sub-clustering plot
    new_disease_dict = {}
    for idx, nd in enumerate(set(sub_clust)):
        new_disease_dict[nd] = idx
    colors3 = [colormap[new_disease_dict[v]] for v in sub_clust]
    _single_plot(subplots, sub_clust, colors3,
                 path.join(outdir,
                           '{0}-sub-clust-plot.png'.format(model_name)))

    _, _, silh = _silhouette_analysis(data_mtx, HCpar['min_cl'],
                                      HCpar['max_cl'], HCpar['linkage_clu'],
                                      HCpar['affinity_clu'])


"""
Private Functions
"""


# clustering evaluation

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

    print('\nAverage Entropy: {0:.2f}'.format(entropy))
    print('Average Purity: {0:.2f}'.format(purity))
    print('Cluster Numerosity:')
    for c in set(clusters):
        print('-- cluster {0}: {1}'.format(c, clusters.count(c)))

    return clusters


def _inner_clustering_analysis(disease_class, data, ehrs, mrns, viz_data,
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
                tmp_raw_ehr.append(ehrs[mrns[idx]])
        print('\nInspecting disease: {0}'.format(dis))
        n_clust, label, _ = _hclust_ehr(
            tmp_data, min_cl, max_cl, linkage, affinity)
        subclass_dis.extend([dis + ': subclust ' + str(l) for l in label])
        list_terms = _freq_term(tmp_raw_ehr, label, vocab, ehrs)
        for l in range(len(set(label))):
            for lt in range(len(list_terms[l])):
                print(
                    'Odds ratio chi2 test for cluster {0}, term: {1}'.format(
                        l, vocab[str(list_terms[l][lt])]))
                try:
                    _chi_test(tmp_raw_ehr, label, list_terms[l][lt], tmp_mrn)
                except ValueError:
                    print('empty class(es)')
            print('')
    return(dis_viz_data, subclass_dis)


def _hclust_ehr(data, min_cl, max_cl, linkage, affinity):
    best_silh = -1
    list_silh = []
    for nc in range(min_cl, max_cl, 1):
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
        print(
            'Number of clusters found: {0}, Silhouette score: {1:.3f}\n'.format(
                n_clust, best_silh))

    except UnboundLocalError:
        hclust = AgglomerativeClustering(n_clusters=min_cl,
                                         linkage=linkage,
                                         affinity=affinity)
        n_clust = min_cl
        label = hclust.fit_predict(data).tolist()
        best_silh = silhouette_score(data, label)
        print(
            'Number of clusters found: {0},',
            'Silhouette score: {1:.3f}\n'.format(n_clust, best_silh))

    return n_clust, label, list_silh


def _chi_test(data, new_classes, term, mrns):
    count_mat = np.zeros((2, len(set(new_classes))), dtype=int)
    for c in set(new_classes):
        for idx, m in enumerate(mrns):
            if new_classes[idx] == c:
                if term in data[idx]:
                    count_mat[1][c] += 1
                else:
                    count_mat[0][c] += 1
    print('Count matrix:\n {0}'.format(count_mat))
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(count_mat)
    print('Chi-squared test statistics:',
          'chi2_stat = {0} -- p_val = {1} -- dof = {2}'.format(
              chi2_stat,
              p_val,
              dof))


def _freq_term(data, pred_class, vocab, raw_ehr):
    list_terms = []
    for subc in range(len(set(pred_class))):
        tmp_data = {}
        for j in range(len(pred_class)):
            if pred_class[j] == subc:
                tmp_data.setdefault(
                    subc, list()).append(
                    [rd for rd in data[j]
                     if rd != 0 and
                     (str.split(vocab[str(rd)], '::')[0]
                      in FRpar['ty_terms'])])
        print('Cluster {0} numerosity: {1}'.format(subc, len(tmp_data[subc])))
        term_count = _freq_dictionary(tmp_data[subc])
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
                print(' '.join(['% most frequent term: {0}',
                                '= {1:.2f} ({2} out of {3} terms in the',
                                'whole dataset -- N patients in',
                                'cluster {4})']).format(
                    vocab[str(MFMT)],
                    subc_termc / num_MFMT,
                    subc_termc,
                    num_MFMT,
                    term_count[MFMT]))
                term_count.pop(MFMT)
                clust_mostfreq.append(MFMT)
            except ValueError:
                pass
        print('')
        list_terms.append(clust_mostfreq)
    return list_terms


def _freq_dictionary(tokens):
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


def _single_plot(data, mrn_disease, colors, name):
    plt.figure(figsize=(20, 10))
    for cl in set(mrn_disease):
        x = [d[0] for j, d in enumerate(data) if mrn_disease[j] == cl]
        y = [d[1] for j, d in enumerate(data) if mrn_disease[j] == cl]
        cols = [c for j, c in enumerate(colors) if mrn_disease[j] == cl]
        plt.xticks([])
        plt.yticks([])
        plt.scatter(x, y, c=cols, label=cl)
    plt.legend(loc=1)
    plt.savefig(name)


# load data

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

    return (mrns, raw_dt, ehrs, vcb)


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


def _silhouette_analysis(data,
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
    print('Running the Silhouette Analysis')
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


# run deep patient model

def _deep_patient(data, n_dim):
    sda = dp.SDA(data.shape[1], nhidden=n_dim, nlayer=3,
                 param={'epochs': 10,
                        'batch_size': 4,
                        'corrupt_lvl': 0.05})
    sda.train(data)
    return sda.apply(data)


# save data

def _save_matrices(datadir, filename, data):
    outfile = path.join(datadir, filename)
    np.save(outfile, data)


def _save_mrns(datadir, data):
    with open(path.join(datadir, 'mrn.txt'), 'w') as f:
        f.write('\n'.join(data))


# miscellaneous

def _print_dataset_statistics(gt_disease, disease_class):
    disease_cnt = {}
    for d in disease_class:
        disease_cnt.setdefault(d, 1)
        disease_cnt[d] += 1
    print('-- no. of subjects: {0}'.format(len(gt_disease)))
    print('-- disease numerosities:')
    for k, v in disease_cnt.items():
        print('----- {0}: {1}'.format(k, v))
    print('')


def _define_colormap():
    col_dict = matplotlib.colors.CSS4_COLORS
    c_out = set(['mintcream', 'cornsilk', 'lavenderblush', 'aliceblue',
                 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige',
                 'powderblue', 'floralwhite', 'ghostwhite', 'lightcoral',
                 'lightcyan', 'lightgoldenrodyellow', 'lightgray',
                 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon',
                 'lightseagreen', 'lightskyblue', 'lightslategray',
                 'lightslategrey', 'lightsteelblue', 'lightyellow', 'linen',
                 'palegoldenrod', 'palegreen', 'paleturquoise',
                 'palevioletred', 'papayawhip', 'peachpuff', 'mistyrose',
                 'lemonchiffon', 'lightblue', 'seashell', 'white',
                 'blanchedalmond', 'oldlace', 'moccasin', 'snow',
                 'darkgray', 'ivory', 'whitesmoke'])
    return [c for c in col_dict if c not in c_out]


"""
Main Function
"""

if __name__ == '__main__':
    print ('')

    # define parameters
    datadir = '../data'
    dt_name = 'ehr-100k'

    indir = os.path.join(datadir, dt_name)
    outdir = os.path.join(datadir, 'experiments',
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
