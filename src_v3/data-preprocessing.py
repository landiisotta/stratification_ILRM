import os
import csv
import re
from time import time
import argparse
import sys
import numpy as np
from utils import dtype
from utils import data_preproc_param as dpp
from load_dataset import load_dataset

'''
Filtering of sequences:
-discard too short sequences;
-cut too long sequences;
-filter least and most frequent terms.
Return the stop word list and the list of:
TERM, sum(PATIENT_FREQUENCY/P), DOCUMENT_FREQUENCY and DF/D*sum(PF/P)
'''


def data_preprocessing(indir,
                       outdir):

    # we don\'t consider icd among the terms to filter out
    # Read vocabulary and EHRs
    pehrs, vidx, pidx = load_dataset(indir)
    
    # experiment folder with date and time to save the representations
    exp_dir = os.path.join(outdir, 'ehr-{0}'.format(len(pehrs.keys())))
    os.makedirs(exp_dir)

    ehr = {}
    ehr_age = {}
    seq_len = []
    for mrn in pehrs.keys():
        ehr_age[mrn] = [list(map(lambda x: int(x), pehrs[mrn]['events'][idx][0:2]))
                        for idx in range(len(pehrs[mrn]['events']))]
        ehr[mrn] = [int(pehrs[mrn]['events'][idx][0])
                    for idx in range(len(pehrs[mrn]['events']))]
        seq_len.append(len(ehr[mrn]))

    f_vocab = {}
    for idx, lab in vidx.items():
        if not(bool(re.match('^icd', lab))):
            f_vocab[lab] = int(idx)

    # Filter most/least frequent terms
    # probability of term w to be in document D and
    # sum of the probabilities of term w to be in patient sequence s
    coll_freq = []
    len_D = len(ehr)
    for l, w in f_vocab.items():
        pws = [ehr[mrn].count(w)/len(ehr[mrn]) for mrn in ehr]
        doc_freq = len(list(filter(lambda x: x != 0, pws)))
        coll_freq.append([l, w, sum(pws), doc_freq])

    # P(w in D)*sum(P(w in s)), where w is the clinical term and s is a patient
    for l in coll_freq:
        tmp = l[2] * l[3]/len_D
        l.append(tmp)
    # sort by the probability product
    coll_freq_sorted = sorted(coll_freq,
                              key=lambda x: x[4])

    # Set minimum threshold
    min_thresh = 1/len_D*(1/np.mean(seq_len))
    max_thresh = 2
    print("Selected terms with score"
          " within [{0:.6f}, {1}]".format(min_thresh, max_thresh))

    stop_words = []
    for cfs in reversed(coll_freq_sorted):
        if cfs[4] < min_thresh or cfs[4] >= max_thresh:
            stop_words.append(int(cfs[1]))

    print("Number of stop words: {0}".format(len(stop_words)))

    # Write stop words file and collection frequencies
    with open(os.path.join(exp_dir, 'stop-words.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["LABEL", "CODE"])
        for s in stop_words:
            wr.writerow([vidx[s], s])
    with open(os.path.join(exp_dir, 'collection-frequencies.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["LABEL", "TERM", "sum(PATIENT_FREQUENCY/P)",
                     "DOCUMENT_FREQUENCY", "DF/D*sum(PF/P)"])
        for el in coll_freq:
            wr.writerow(el)

    # Sort ehrs with respect to age_in_days and eliminate stop words
    # Check min sequence length
    ehr_rid = {}
    count_short = 0
    count_long = 0
    for mrn in ehr_age:
        if len(ehr_age[mrn]) > dpp['min_seq_len']:
            ehr_rid.setdefault(mrn,
                               list()).extend(list(filter(lambda x:
                                                          x[0] not in stop_words,
                                                          ehr_age[mrn])))
            ehr_rid[mrn] = sorted(ehr_rid[mrn], key=lambda x: x[1])
        else:
            count_short += 1

    # Set and shuffle
    ehr_subseq = {}
    subseq_len = 0
    n_subseq = 0
    for mrn, vect in ehr_rid.items():
        step = vect[0][1] + dpp['age_step']
        tmp = [vect[0]]
        term_track = set([vect[0][0]])
        for el in vect[1::]:
            if el[1] <= step:
                if el[0] not in term_track:
                    tmp.append(el)
                    term_track.add(el[0])
            else:
                subseq_len += len(tmp)
                n_subseq += 1
                np.random.shuffle(tmp)
                ehr_subseq.setdefault(mrn, list()).extend(tmp)
                step = el[1] + dpp['age_step']
                tmp = [el]
                term_track = set([el[0]])
        subseq_len += len(tmp)
        n_subseq += 1
        np.random.shuffle(tmp)
        ehr_subseq.setdefault(mrn, list()).extend(tmp)
        # Remove too short sequences and trim too long sequences
        if len(ehr_subseq[mrn]) > dpp['max_seq_len']:
            count_long += 1
            ehr_subseq[mrn] = ehr_subseq[mrn][-dpp['max_seq_len']:]
        elif len(ehr_subseq[mrn]) < dpp['min_seq_len']:
            count_short += 1
            ehr_subseq.pop(mrn)

    # Report statistics
    print("The average number of tokens for each time slot of {0} days is:"
          " {1:.3f}".format(dpp['age_step'], subseq_len/n_subseq))
    print("Number of dropped sequences: {0} (< {1}),"
          " number of trimmed sequences: {2} (> {3}).\n"
          "Initial total number of patients: {4}.\n"
          "Current number of patients: {5}".format(count_short,
                                                   dpp['min_seq_len'],
                                                   count_long,
                                                   dpp['max_seq_len'],
                                                   len(ehr_age),
                                                   len(ehr_subseq)))
    f_seq_len = []
    for mrn in ehr_subseq:
        f_seq_len.append(len(ehr_subseq[mrn]))
    print("The average length of ehr sequences is:"
          " {0:.2f}".format(np.mean(f_seq_len)))
    print("The sequence length ranges"
          " from {0} to {1}".format(min(f_seq_len),
                                    max(f_seq_len)))

    # Write ehrs 1)[MRN, EHRseq]; 2)[MRN,AID_start,AID_end,EHRseq]
    with open(os.path.join(exp_dir, 'cohort-new-ehrseq.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "EHRseq"])
        for mrn in ehr_subseq:
            wr.writerow([mrn] + [e[0] for e in ehr_subseq[mrn]])

    with open(os.path.join(exp_dir, 'cohort-new-ehrseq-age_in_day.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "AID_start", "AID_stop", "EHRseq"])
        for mrn in ehr_subseq:
            wr.writerow([mrn, ehr_subseq[mrn][0][1],
                         ehr_subseq[mrn][-1][1]] + [e[0] for e in ehr_subseq[mrn]])

    with open(os.path.join(exp_dir, 'cohort-new-ehr-age_in_day.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "AID", "EHR"])
        for mrn, el in ehr_subseq.items():
            for e in el:
                wr.writerow([mrn, e[1], e[0]])

    # Create and save new vocabulary starting from 0
    tdx = 0
    new_vocab = {}
    for idx, lab in vidx.items():
        if idx not in stop_words:
            new_vocab[vidx[idx]] = tdx
            tdx += 1
    print("Dropped {0} out of {1} terms. Current"
          " number of terms is: {2}".format((len(vidx)-len(new_vocab)),
                                            len(vidx),
                                            len(new_vocab)))

    with open(os.path.join(exp_dir, 'cohort-new-vocab.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['LABEL', 'CODE'])
        for l, c in new_vocab.items():
            wr.writerow([l, c])

# main function


def _process_args():
    parser = argparse.ArgumentParser(
             description='EHR Preprocessing')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument(dest='outdir', help='Output directory')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print('')

    start = time()
    data_preprocessing(indir=args.indir,
                       outdir=args.outdir)

    print('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print('Task completed\n')
