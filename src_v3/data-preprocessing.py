import os
import csv
import re
from time import time
import argparse
import sys
import numpy as np
from utils import f_dtype
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

def subseq_set_shuffle(data):
    # Set and shuffle
    ehr_subseq = {}
    subseq_len = 0
    n_subseq = 0
    tmp_count_long = 0
    tmp_count_short = 0
    for mrn, vect in data.items():
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
            tmp_count_long += 1
            ehr_subseq[mrn] = ehr_subseq[mrn][-dpp['max_seq_len']:]
        elif len(ehr_subseq[mrn]) < dpp['min_seq_len']:
            tmp_count_short += 1
            ehr_subseq.pop(mrn)

    print("The average number of tokens for each time slot of {0} days is:"
          " {1:.3f}\n".format(dpp['age_step'], subseq_len/n_subseq))

    return ehr_subseq, tmp_count_short, tmp_count_long


def data_preprocessing(indir,
                       outdir,
                       test_set=None):

    # we don\'t consider icd among the terms to filter out
    # Read vocabulary and EHRs
    pehrs, vidx, pidx = load_dataset(indir)

    count_short = 0
    count_long = 0
   
    if test_set is not None:
        with open(os.path.join(indir, test_set)) as f:
            rd = csv.reader(f)
            mrn_test = [r[0] for r in rd if r[0] in pehrs.keys()] # ELIMINATE IF 
        pehrs_ts = {mrn: pehrs[mrn] for mrn in mrn_test}
        pehrs_tr = {mrn: pehrs[mrn] for mrn in pehrs.keys() 
                    if mrn not in mrn_test}
        pehrs = pehrs_tr

        # experiment folder with date and time to save the representations
        exp_dir = os.path.join(outdir, 'ehr-{0}'.format('-'.join([str(len(pehrs.keys())), 
                                                        test_set.split('.')[0]])))
    else:
        exp_dir = os.path.join(outdir, 'ehr-{0}'.format(len(pehrs.keys())))

    os.makedirs(exp_dir)

    # filter terms from utils (f_dtype)           
    vocab_tmp = {}
    f_terms = []
    for idx, lab in vidx.items():
        if not(bool(re.match('|'.join(list(map(lambda x: '^' + x, f_dtype))), lab))):
            vocab_tmp[idx] = lab
        else:
            f_terms.append(lab)
    vidx = vocab_tmp
        
    ehr = {}
    ehr_age = {}
    seq_len = []
    for mrn in pehrs.keys():
        tmp_list = list(filter(lambda x: x[0] not in f_terms,
                               [pehrs[mrn]['events'][idx][0:2]
                               for idx in range(len(pehrs[mrn]['events']))]))
        if len(tmp_list) > dpp['min_seq_len']:
            ehr_age[mrn] = tmp_list
            ehr[mrn] = [tl[0] for tl in tmp_list]
            seq_len.append(len(ehr[mrn]))
        else:
            count_short += 1
    print("Dropped {0} terms. Dtype: {1}\n".format(len(f_terms), '-'.join(f_dtype)))

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
        pws = [ehr[mrn].count(l)/len(ehr[mrn]) for mrn in ehr]
        doc_freq = len(list(filter(lambda x: x != 0, pws)))
        coll_freq.append([l, sum(pws), doc_freq])

    # P(w in D)*sum(P(w in s)), where w is the clinical term and s is a patient
    for cf in coll_freq:
        tmp = cf[1] * cf[2]/len_D
        cf.append(tmp)
    # sort by the probability product
    coll_freq_sorted = sorted(coll_freq,
                              key=lambda x: x[3])

    # Set minimum threshold
    min_thresh = 0.001
    max_thresh = 0.2
    print("Selected terms with score"
          " within [{0:.6f}, {1}]".format(min_thresh, max_thresh))

    stop_words = []
    for cfs in reversed(coll_freq_sorted):
        if cfs[3] <= min_thresh or cfs[3] >= max_thresh:
            stop_words.append(cfs[0])
    for c in coll_freq_sorted: 
        if (c[-1] < min_thresh and c[-1]!=0) or c[-1] > max_thresh:
            label = c[0].split('::')
            print([label[0], label[1]] + c[1::])
    print('\n')
    print("Number of stop words: {0}".format(len(stop_words)))
    # Write stop words file and collection frequencies
    with open(os.path.join(exp_dir, 'stop-words.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["LABEL"])
        for s in stop_words:
            wr.writerow([s])
    with open(os.path.join(exp_dir, 'collection-frequencies.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["LABEL", "sum(PATIENT_FREQUENCY/P)",
                     "DOCUMENT_FREQUENCY", "DF/D*sum(PF/P)"])
        wr.writerows([el for el in coll_freq_sorted])

    # Sort ehrs with respect to age_in_days and eliminate stop words
    # Check min sequence length
    ehr_rid = {}
    for mrn in ehr_age:
        tmp_list = list(filter(lambda x:
                               x[0] not in stop_words,
                               ehr_age[mrn]))
        if len(tmp_list) > dpp['min_seq_len']:
            ehr_rid.setdefault(mrn,list()).extend(sorted(tmp_list, key=lambda x: x[1]))
        else:
            count_short += 1
   
    # create new vocabulary
    with open(os.path.join(exp_dir, 'cohort-new-vocab.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['LABEL', 'CODE'])
        cod = 0
        new_vidx = {}
        for idx, lab in vidx.items():
            if lab not in stop_words:
                wr.writerow([lab, cod])
                new_vidx[lab] = int(cod)
                cod += 1
 
    if test_set is not None:  
        ehr_rid_ts = {}
        for mrn in pehrs_ts.keys():
            tmp_list = list(filter(lambda x: x[0] not in f_terms + stop_words,
                                   [pehrs_ts[mrn]['events'][idx][0:2]
                                   for idx in range(len(pehrs_ts[mrn]['events']))]))
        if len(tmp_list) > dpp['min_seq_len']:
            ehr_rid_ts.setdefault(mrn, list()).extend(sorted(tmp_list, 
                                                      key=lambda x: x[1]))
        print("Set and shuffle subsequences for test set:")
        ehr_subseq_ts, _, _ = subseq_set_shuffle(ehr_rid_ts)

        # Write ehrs 1)[MRN, EHRseq]; 2)[MRN,AID_start,AID_end,EHRseq]
        with open(os.path.join(exp_dir, 'cohort_test-new-ehrseq.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(["MRN", "EHRseq"])
            for mrn in ehr_subseq_ts:
                wr.writerow([mrn] + [new_vidx[e[0]] for e in ehr_subseq_ts[mrn]])

        with open(os.path.join(exp_dir, 'cohort_test-new-ehrseq-age_in_day.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(["MRN", "AID_start", "AID_stop", "EHRseq"])
            for mrn in ehr_subseq_ts:
                wr.writerow([mrn, ehr_subseq_ts[mrn][0][1],
                             ehr_subseq_ts[mrn][-1][1]] + [new_vidx[e[0]] for e in ehr_subseq_ts[mrn]])

        with open(os.path.join(exp_dir, 'cohort_test-new-ehr-age_in_day.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(["MRN", "AID", "EHR"])
            for mrn, el in ehr_subseq_ts.items():
                for e in el:
                    wr.writerow([mrn, e[1], new_vidx[e[0]]])


    ehr_subseq, tmp_count_short, tmp_count_long = subseq_set_shuffle(ehr_rid)
    count_short += tmp_count_short
    count_long += tmp_count_long

    # Report statistics
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


    # Create and save new vocabulary without stop words
    print("Dropped {0} out of {1} terms. Current"
          " number of terms is: {2}".format(len(stop_words)+len(f_terms),
                                            len(vidx)+len(f_terms),
                                            len(vidx)-len(stop_words)))

    # Write ehrs 1)[MRN, EHRseq]; 2)[MRN,AID_start,AID_end,EHRseq]
    with open(os.path.join(exp_dir, 'cohort-new-ehrseq.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "EHRseq"])
        for mrn in ehr_subseq:
            wr.writerow([mrn] + [new_vidx[e[0]] for e in ehr_subseq[mrn]])

    with open(os.path.join(exp_dir, 'cohort-new-ehrseq-age_in_day.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "AID_start", "AID_stop", "EHRseq"])
        for mrn in ehr_subseq:
            wr.writerow([mrn, ehr_subseq[mrn][0][1],
                         ehr_subseq[mrn][-1][1]] + [new_vidx[e[0]] for e in ehr_subseq[mrn]])

    with open(os.path.join(exp_dir, 'cohort-new-ehr-age_in_day.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "AID", "EHR"])
        for mrn, el in ehr_subseq.items():
            for e in el:
                wr.writerow([mrn, e[1], new_vidx[e[0]]])

# main function


def _process_args():
    parser = argparse.ArgumentParser(
             description='EHR Preprocessing')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument(dest='outdir', help='Output directory')
    parser.add_argument(dest='test_set', default=None, 
                        type=str, help='Add fold')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print('')

    start = time()
    data_preprocessing(indir=args.indir,
                       outdir=args.outdir,
                       test_set=args.test_set)
                       
    print('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print('Task completed\n')
