'''
Filtering of sequences:
-discard too short sequences;
-cut too long sequences;
-filter least and most frequent terms.
Return the stop word list and the list of TERM, sum(PATIENT_FREQUENCY/P), DOCUMENT_FREQUENCY and DF/D*sum(PF/P)
'''

import os
import csv
import re
from utils import dtype
from utils import data_preprocessing_pars as dpp

##we don't consider icd among the terms to filte out
with open(os.path.join(outdir, 'cohort-vocab.csv')) as f:
    rd = csv.reader(f)
    next(rd)
    vocab = {}
    f_vocab = {}
    for r in rd:
        if not(bool(re.match('^icd', r))):
            f_vocab[r[0]] = int(r[1])
        vocab[r[0]] = int(r[1])

with open(os.path.join(outdir, 'cohort-ehr.csv')) as f:
    rd = csv.reader(f)
    next(rd)
    ehr_age = {}
    for r in rd:
        ehr_age.setdefault(r[0], list()).append((int(r[1]), int(r[2])))

##probability of term w to be in document D and 
##sum of the probabilities of term w to be in patient sequence s
coll_freq = []
len_D = len(ehr_age)
for l, w in f_vocab.items():
    pws = [ehr_age[m].count(w)/len(ehr_age[m]) for m in ehr_age]
    doc_freq = len(list(filter(lambda x: x!=0, pws)))
    coll_freq.append([l, w, sum(pws), doc_freq])

##P(w in D)*sum(P(w in s)), where w is the clinical term and s is a patient
for l in coll_freq:
    tmp = l[2] * l[3]/len_D
    l.append(tmp)
coll_freq_sorted = sorted(coll_freq, key=lambda x: x[4]) ##sort by the probability product

##Set minimum threshold 
thresh = 0.002
stop_words = []
for cfs in reversed(coll_freq_sorted):
    if cfs[4] >= thresh or cfs[4] <= (1/len_D*sum([1/len(ehr_age[m]) for m in ehr_age])):
        stop_words.append([cfs[0], cfs[1]])

print("Number of stop words: {0}".format(len(stop_words)))

with open(os.path.join(outdir, 'stop-words.csv'), 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_NONE)
    wr.writerow(["LABEL", "CODE"])
    wr.writerows([s for s in stop_words])
with open(os.path.join(outdir, 'collection-frequencies.csv'), 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_NONE)
    wr.writerow(["LABEL", "TERM", "sum(PATIENT_FREQUENCY/P)", 
                 "DOCUMENT_FREQUENCY", "DF/D*sum(PF/P)"])
    for el in coll_freq:
        wr.writerow(el)

##Sort ehrs with respect to age_in_days and eliminate stop words
##Check min sequence length
ehr_rid = {}
count_short = 0
count_long = 0
for m in ehr_age:
    if len(ehr_age[m]) > dpp['min_seq_len']:  
        ehr_rid.setdefault(m, list()).extend(list(filter(lambda x: x[0] not in stop_words, 
                                                         ehr_age[m])))
        ehr_rid[m] = sorted(ehr_rid[m], key=lambda x: x[1])
    else:
        count_short += 1

##Set and shuffle 
ehr_subseq = {}
subseq_len = 0
n_subseq = 0
for m in ehr_rid:
    step = ehr_rid[m][0][1] + dpp['age_step']
    tmp = []
    for el in ehr_rid[m]:
        if el[1] <= step:
            if el[0] not in tmp:
                tmp.append(el))
        else:
            subseq_len += len(tmp)
            n_subseq += 1
            ehr_subseq.setdefault(m, list()).extend(np.random.shuffle(tmp))
            tmp = []
            step = el[1] + dpp['age_step']
            tmp.append(el)
    subseq_len += len(tmp)
    n_subseq += 1 
    ehr_subseq.setdefault(m, list()).extend(np.random.shuffle(tmp))
    if len(ehr_subseq[m]) > dpp['max_seq_len']:
        count_long += 1
        ehr_subseq[m] = ehr_subseq[m][-dpp['max_seq_len']:]
    elif len(ehr_subseq[m]) < dpp['min_seq_len']:
        count_short += 1
        ehr_subseq.pop(m)

print("The average number of tokens for each time slot of {0} days is: \
       {1:.3f%}".format(dpp['age_step'], subseq_len/n_subseq))
print("Number of dropped sequences: {0} (< {1}), number of trimmed sequences: {2} (> {3}). \
       Initial total number of sequences: {4}.".format(count_short, dpp['min_seq_len'],
                                                       count_long, dpp['max_seq_len'],
                                                       len(ehr_age)))

l = []
    for m in ehr_subseq:
        l.append(len(ehr_subseq[m]))
print("The average length of ehr sequences is: {0:.2f}".format(np.mean(l)))
print("The sequence length ranges from {0} to {1}".format(min(l), max(l)))

with open(os.path.join(outdir, 'cohort-ehr-shuffle.csv'), 'w') as f:
    wr = csv.writer(f)
    for m in ehr_subseq:
        wr.writerow([m] + [e[0] for e in ehr_subseq[m]])

with open(os.path.join(outdir, 'cohort-ehr-shuffle-age.csv'), 'w') as f:
    wr = csv.writer(f)
    for m in ehr_subseq:
        wr.writerow([m, ehr_subseq[m][0][1], ehr_subseq[m][-1][1]] + [e[0] for e in ehr_subseq[m]])

