import utils as ut
import sys
from time import time
import utils
from sklearn.cluster import AgglomerativeClustering
import os
import csv
import argparse
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.metrics import silhouette_score
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests
from itertools import combinations
import itertools

#Input: ehr lists corresponding to a cluster 
#Output: dictionary of relative term counts
def FreqDict(data, subc, vocab):
    set_data = list(data.values())
    mrgd_data = list(itertools.chain.from_iterable(set_data))
    subc_vect = data[subc]
    unq_mrgd, cnt_mrgd = np.unique(mrgd_data, 
                                   return_counts=True)
    unq_subc, cnt_subc = np.unique(subc_vect, 
                                   return_counts=True)
    all_freq = dict(zip(unq_mrgd, cnt_mrgd))
    freq_dict = dict(zip(unq_subc, cnt_subc))
    new_dict = {t: (freq_dict[t] / all_freq[t],
                     freq_dict[t], all_freq[t])
                for t in freq_dict 
                if vocab[t].split('::')[0] in ut.select_terms}
    return new_dict


#Input: dictionary cluster:ehrs; list mrns
#Output:
def freq_term(data, pred_class, vocab):
    list_terms = []
    tmp_data = {}
    set_data = {}
    for mrn, subc in pred_class.items():
        tmp_data.setdefault(str(subc), list()).append(data[mrn])
        set_data.setdefault(str(subc), list()).extend(list(set(data[mrn])))
    for subc in range(len(set(tmp_data.keys()))):
        print("Cluster {0} numerosity: {1}".format(subc, len(tmp_data[str(subc)])))
        # percentage of term in subc wrt disease cohort
        rel_term_count = FreqDict(set_data, str(subc), vocab)
        clust_mostfreq = []
        for l in range(ut.FRpar['n_terms']):
            try:
                MFMT = max(rel_term_count, 
                           key=(lambda key: rel_term_count[key][1]))
                subc_termc = rel_term_count[MFMT][1]
                num_MFMT = rel_term_count[MFMT][2]
                print("% term:{0} "
                      "= {1:.2f} ({2} terms in cluster ({3:.2f}), "
                      "out of {4} terms in the disease dataset\n".format(vocab[str(MFMT)], 
                                                             rel_term_count[MFMT][0], 
                                                             subc_termc,
                                                             subc_termc/len(tmp_data[str(subc)]),
                                                             num_MFMT))
                rel_term_count.pop(MFMT)
                clust_mostfreq.append(MFMT)
            except ValueError:
                pass
        print("\n")
        list_terms.append(clust_mostfreq)
        for lt in range(len(list_terms[subc])):
            try:
                print("Pairwise odds ratio chi2 test for "
                      "term: {0}".format(vocab[str(list_terms[subc][lt])]))
                chi_test(raw_ehr, pred_class, list_terms[subc][lt])
            except ValueError:
                pass
        print("\n\n")
    return list_terms


##Hierarchical clustering function for outer validation. Max silhouette.
def hclust_ehr(data, min_cl, max_cl, linkage, affinity):
    best_silh = -1
    list_silh = []
    mrns = list(data.keys())
    data = np.array(list(data.values())).astype(np.float32)
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
        print("Number of clusters found:{0}, "
               "Silhouette score:{1:.3f}\n".format(n_clust, best_silh))
    except UnboundLocalError:
        hclust = AgglomerativeClustering(n_clusters=min_cl,
                                         linkage=linkage,
                                         affinity=affinity)
        n_clust = min_cl
        label = hclust.fit_predict(data).tolist()
        best_silh = silhouette_score(data, label)
        print("Number of clusters found:{0}, "
              "Silhouette score:{1:.3f}\n".format(n_clust, best_silh))
    return n_clust, {m: l for m, l in zip(mrns, label)}, list_silh


# pairwise chi-square test with bonferroni correction
# print only significant comaprisons
def chi_test(data, new_classes, term):
    subc_vect = []
    yes = []
    no = []
    for mrn, subc in new_classes.items():
        subc_vect.append(subc)
        yes.append(list(set(data[mrn])).count(term))
    no = list(map(lambda x: 1-x, yes))
    df = pd.DataFrame()
    df['subcluster'] = subc_vect
    df['0'] = no
    df['1'] = yes
    sum_df = df.groupby(['subcluster']).sum()
    all_comb = list(combinations(sum_df.index, 2))
    p_vals = []
    for comb in all_comb:
        new_df = sum_df[(sum_df.index == comb[0]) | (sum_df.index == comb[1])]
        chi2, pval, _, _ = chi2_contingency(new_df, correction=True)
        p_vals.append(pval)
    reject_list, corrected_p_vals = multipletests(p_vals, method='bonferroni')[:2]
    for comb, pv, cpv, r in zip(all_comb, p_vals, corrected_p_vals, reject_list):
        print("Comparison: {0} -- p={1}, corr_p={2}, rej={3}".format(
              comb, pv, cpv, r))
    print(sum_df)


# pairwise ttest for confounders age and seq_len
def pairwise_ttest(val_vec, cnf):
    df = pd.DataFrame()
    cluster = []
    score = []
    for subc, dic_conf in val_vec.items():
        cluster += [str(subc) for idx in range(len(dic_conf[cnf]))]
        score.extend(dic_conf[cnf])
    df['subcluster'] = cluster
    df['score'] = score
#    all_comb = list(combinations(df.subcluster, 2))
#    p_vals = []
#    for comb in all_comb:
#        g1 = df[(df.subcluster == comb[0])]['score'] 
#        g2 = df[(df.subcluster == comb[1])]['score']
#        stat, pval = ttest_ind(g1, g2, equal_var=False)
#        p_vals.append(pval)
#    reject_list, corrected_p_vals = multipletests(p_vals, method='bonferroni')[:2]
#    for comb, pv, cpv, r in zip(all_comb, p_vals, corrected_p_vals, reject_list):
#        print("Comparison: {0} -- p={1}, corr_p={2}, rej={3}".format(
#              comb, pv, cpv, r))
    MultiComp = MultiComparison(df['score'],
                                df['subcluster'])
    comp = MultiComp.allpairtest(ttest_ind, 
                                 method='bonf')
    print(comp[0])
    pd.options.display.float_format = '{:.3f}'.format
    print(df.groupby(['subcluster']).describe())


# pairwise chi-sq test for confounder sex
def pairwise_chisq(val_vec, cnf):
    cluster = list(val_vec.keys())
    fem = [val_vec[subc][cnf].count('Female') 
           for subc in val_vec.keys()]
    mal = [val_vec[subc][cnf].count('Male')
           for subc in val_vec.keys()]
    df = pd.DataFrame()
    df['female'] = fem
    df['male'] = mal
    df.index = cluster
    all_comb = list(combinations(df.index, 2))
    p_vals = []
    for comb in all_comb:
        new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
        chi2, pval, _, _ = chi2_contingency(new_df, correction=True)
        p_vals.append(pval)
    reject_list, corrected_p_vals = multipletests(p_vals, method='bonferroni')[:2]
    for comb, pv, cpv, r in zip(all_comb, p_vals, corrected_p_vals, reject_list):
        print("Comparison: {0} -- p={1:.3f}, corr_p={2:.3f}, rej={3}".format(
              comb, pv, cpv, r))
    print(df)


# Check for confounders between subclusters (i.e. SEX, AGE, SEQ_LEN)
def check_confounders(datadir, raw_ehr, label):
    with open(os.path.join(datadir, 'patient-details.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        p_details = {}
        for r in rd:
            if r[0] in list(label.keys()):
                p_details[r[0]] = [float(r[1]), r[2], len(raw_ehr[r[0]])]
    n_cl = len(set(label.values()))
    val_vec = {}
    for mrn, subc in label.items():
        if subc in val_vec:
            val_vec[subc]['age'].append(p_details[mrn][0])
            val_vec[subc]['sex'].append(p_details[mrn][1])
            val_vec[subc]['seq_len'].append(p_details[mrn][2])
        else:
            val_vec[subc] = {'age':[p_details[mrn][0]],
                             'sex': [p_details[mrn][1]],
                             'seq_len': [p_details[mrn][2]]}
    print("Multiple comparison t-test for AGE:")
    pairwise_ttest(val_vec, 'age')
    print("\nMultiple comparison t-test for SEQUENCE LENGTH:")
    pairwise_ttest(val_vec, 'seq_len')
    print("\nMultiple comparison chi-square test for SEX:")
    pairwise_chisq(val_vec, 'sex')


##Internal clustering validation
"""
data: convae output (mrn, avg_vect)
raw_ehr: (mrn, term_seq)
mrn_dis: [m for m in mrn with disease]
"""
def inner_clustering_analysis(disease, data, raw_ehr,
                              min_cl, max_cl, linkage, affinity, 
                              vocab):
    print("Cohort {0} numerosity: {1}\n".format(disease, len(data)))
    n_clust, mrn_label, _ = hclust_ehr(data, min_cl, max_cl, linkage, affinity)
    freq_term(raw_ehr, mrn_label, vocab)
    return mrn_label


"""
Main function
"""


def _process_args():

    parser = argparse.ArgumentParser(
          description='Inner clustering encodings evaluation')
    parser.add_argument(dest='datadir',
          help='Directory where disease are stored')
    parser.add_argument(dest='indir',
          help='Directory with complete ehrs')
    parser.add_argument(dest='encdir_test1',
          help='Directory with encodings for test1')
    parser.add_argument(dest='encdir_test2',
          help='Directory with encodings for test2')
    parser.add_argument(dest='disease',
          help='Specify disease acronym')

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = _process_args()
    print('')
    start = time()

    with open(os.path.join(args.datadir, 
                           'cohort-{0}.csv'.format(args.disease))) as f:
        rd = csv.reader(f)
        mrns_dis = [str(r[0]) for r in rd]
    with open(os.path.join(args.encdir_test1, 
                           'convae-avg_vect.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        data = {}
        for r in rd:
            if r[0] in mrns_dis:
                data[r[0]] = r[1::]
    with open(os.path.join(args.encdir_test2,
                           'convae-avg_vect.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r[0] in mrns_dis:
                data[r[0]] = r[1::]
    with open(os.path.join(args.indir, 
                           'cohort-new-ehrseq.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        raw_ehr = {}
        for r in rd:
            if r[0] in mrns_dis:
                raw_ehr[r[0]] = r[1::]
    with open(os.path.join(args.indir, 
                           'cohort-new-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        idx_to_mt = {r[1]: r[0] for r in rd}

    label = inner_clustering_analysis(args.disease, data, 
                                      raw_ehr, 
                                      ut.HCpar['min_cl'],
                                      ut.HCpar['max_cl'], 
                                      ut.HCpar['linkage_clu'],
                                      ut.HCpar['affinity_clu'],
                                      idx_to_mt)

    with open(os.path.join(args.datadir, 
         'cohort-{}-innerval-labels.csv'.format(args.disease)), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "OUT-CL"])
        for mrn, lab in label.items():
            wr.writerow([mrn, lab])

    check_confounders(args.datadir, raw_ehr, label)

    print('\nProcessing time: %s seconds' % round(time()-start, 2))
    print('Task completed\n')

