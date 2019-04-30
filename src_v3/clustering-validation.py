from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
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

    # normalize data
    if preproc:
        data = preprocessing.scale(data)

    # external cluster analysis
    _outer_clustering_analysis(data, gt_clu, affinity_clu)

    return

# external clustering analysis
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
    gt_file = os.path.join(indir, ut.dt_files['diseases'])
    gt_disease = clu.load_mrn_disease(gt_file)
    min_clu = 2
    max_clu = 10

    print('\nRunning clustering on the encoded vectors')
    gt_disease_enc = [gt_disease[m][0] for m in mrn]
    clu.eval_hierarchical_clustering(
        encoded, gt_disease_enc, min_clu, max_clu, preproc=True)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 2))
