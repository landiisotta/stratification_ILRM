from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import numpy as np
import csv
import utils as ut
from utils import HCpar
import re
import argparse
from time import time

def clustering_inspection(datadir, indir, outdir, list_model):
    
    with open(os.path.join(indir, 'cohort-new-vocab.csv')) as f:
        vocab = {}
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            vocab[r[0]] = r[1]

    print('Loading data...')
    baselines = {}
    mrn = False
    for mod in list_model:
        if mod == 'conv-ae':
            with open(os.path.join(outdir, mod + '-avg*')) as f:
                rd = csv.reader(f)
                convae_mrn = convae_v = []
                for r in rd:
                convae_mtx.append(r[1::])
                convae_mrn.append([r[0])
            gt_disease_convae, disease_class_convae = _load_ground_truth(datadir, 
                                                                         convae_mrn)
            cluster_evaluation(convae_mtx, convae_mrn, 
                               disease_class_convae, vocab, mod)
        else:
            baseline_mtx = np.load(os.path.join(outdir, mod + '*'))
            if mrn:
                with open(os.path.join(outdir, 'bs-mrn.txt')) as f:
                    rd = csv.reader(f)
                    mrn_list = [r for r in rd]
                gt_disease_bs, disease_class_bs = _load_groud_truth(datadir, 
                                                                    mrn_list)
            mrn = True
            cluster_evaluation(baseline_mtx, disease_class, mrn_list, vocab, mod)


def cluster_evaluation(data_mtx, gt_clu, mrns, vocab,
                       HCpar, model_name):

    label = model_name.upper()
    print('Evaluating {0} embeddings\n'.format(label))
    
    # evaluate cluster results
    clu = _outer_clustering_analysis(
          data_mtx, gt_clu, 
          HCpar['linkage_clu'], HCpar['affinity_clu'])

    with open(os.path.join(outdir, 'outer-cl-{0}.txt'.format(model_name)), 'w') as f:
        f.write('\n'.join(clu))

"""
Private Functions
"""

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


def _load_ground_truth(indir, mrns):

    # list of diagnosed diseases associated with MRNs
    with open(path.join(indir, 'cohort-mrn_diseases.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        mrn_disease = {}
        for r in rd:
            if r[0] not in mrn_diseases and bool(re.match('|'.join(ut.diseases), r[1].lower())):
                mrn_disease[r[0]] = r[1]

    # ground truth using the first diagnosis
    gt_disease = {m: mrn_disease[m][0]
                  for m in mrn_disease if m in mrn_set}

    # define disease class
    disease_class = [gt_disease[m] for m in mrns]

    return (gt_disease, disease_class)


"""
Main Function
"""
def _process_args():
    parser = argparse.ArgumentParser(
          description='Outer clustering encodings evaluation')
    parser.add_argument(dest='datadir', 
          help='Directory whith the unprocessed data')
    parser.add_argument(dest='indir',
          help='Directory with the processed data')
    parser.add_argument(dest='outdir',
          help='Directory with the encodings')


if __name__ == '__main__':
    args = _process_args()
    print('')

    start = time()
    
    clustering_inspection(datadir=args.datadir,
                          indir=args.indir,
                          outdir=args.outdir,
                          list_model=ut.list_model)

    print('\nProcessing time: %s seconds' % round(time()-start, 2))

    print('Task completed\n') 
