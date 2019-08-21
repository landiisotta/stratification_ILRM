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

# computationally expensive function here! (see log file for partial times and profile) 
def COHORTsubseq_set_shuffle(data, stop_words):
    len_ss = n_ss = 0
    tmp_count_short = tmp_count_long = 0
    ehr_subseq = {}
    for idx, (mrn, vect) in enumerate(data.items()):
        idx = jdx = 0
        ss = []
        while idx < len(vect):
            ss_tmp = []
            term_tmp = set()
            while jdx < len(vect) and (vect[jdx][1] - vect[idx][1]) <= dpp['age_step']:
                if vect[jdx][0] not in term_tmp and vect[jdx][0] not in stop_words:
                    ss_tmp.append(vect[jdx])
                    term_tmp.add(vect[jdx][0])
                jdx += 1
            np.random.shuffle(ss_tmp)
            ss.extend(ss_tmp)
            len_ss += len(ss_tmp)
            n_ss += 1
            idx = jdx
        if len(ss) < dpp['min_seq_len']:
            tmp_count_short += 1
        elif len(ss) > dpp['max_seq_len']:
            ehr_subseq.setdefault(mrn, 
                                  list()).extend(ss[-dpp['max_seq_len']:])
            tmp_count_long += 1
        else:
            ehr_subseq.setdefault(mrn, 
                                  list()).extend(ss)
    print("The average number of tokens for each time slot of {0} days is:"
          " {1:.3f}".format(dpp['age_step'], len_ss/n_ss))
    print("The average number of subsequences for each "
          "subject is: {0:.3f}".format(n_ss/len(data)))
    
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
        print("Loading test set: {0}".format(test_set))
        with open(os.path.join(indir, test_set)) as f:
            rd = csv.reader(f)
            mrn_test = [r[0] for r in rd if r[0] in pehrs.keys()] # NO NEED FOR THIS FOR 1M DATA 
        pehrs_ts = {mrn: pehrs[mrn] for mrn in mrn_test}
        pehrs_tr = {mrn: pehrs[mrn] for mrn in pehrs.keys() 
                    if mrn not in mrn_test}
        pehrs = pehrs_tr
        print("Loaded test data for {0} patients\n".format(len(pehrs_ts)))
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
    for mrn in pehrs.keys():
        tmp_list = list(filter(lambda x: x[0] not in f_terms,
                               [pehrs[mrn]['events'][idx][0:2]
                               for idx in range(len(pehrs[mrn]['events']))]))
        if len(tmp_list) > dpp['min_seq_len']:
            ehr_age[mrn] = tmp_list
            ehr[mrn] = [tl[0] for tl in tmp_list]
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
    count_dict = dict.fromkeys(f_vocab.keys(), [0, 0])
    for mrn in ehr:
        ehr_a = np.array(ehr[mrn])        
        unique, t_count = np.unique(ehr_a, return_counts=True)
        t_count = t_count/len(ehr_a)
        tmp_dict = dict(zip(unique, t_count))
        for t, c in tmp_dict.items():
            if t in f_vocab.keys() and c !=0:
                count_dict[t] = list(map(sum, zip(count_dict[t], [c, 1])))    

    # P(w in D)*sum(P(w in s)), where w is the clinical term and s is a patient
    coll_freq = [[t] + c[::] + [c[0] * c[1]/len_D] 
                 for t, c in count_dict.items()]
    # sort by the probability product
    coll_freq_sorted = sorted(coll_freq,
                              key=lambda x: x[3])

    # Set minimum threshold
    min_thresh = 1e-07
    print("Selected terms with score"
          " greater than {0}".format(min_thresh))

    stop_words = []
    for cfs in reversed(coll_freq_sorted):
        if cfs[-1] <= min_thresh:
            stop_words.append(cfs[0])
            if cfs[-1] !=0:
                label = cfs[0].split('::')
                print([label[0], label[1]] + cfs[1::])
    
    print('\n')
    print("Number of stop words: {0}".format(len(stop_words)))
    # Write stop words file and collection frequencies
    with open(os.path.join(exp_dir, 'stop-words.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["LABEL"])
        wr.writerows([s for s in stop_words])
    with open(os.path.join(exp_dir, 'collection-frequencies.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["LABEL", "sum(PATIENT_FREQUENCY/P)",
                     "DOCUMENT_FREQUENCY", "DF/D*sum(PF/P)"])
        wr.writerows([el for el in coll_freq_sorted])

    print('End stop words selection. Time: {0} seconds\n'.format(round(time() - start, 2)))

    # Sort ehrs with respect to age_in_days and eliminate stop words
    # Check min sequence length
    #ehr_rid = {}
    #for mrn in ehr_age:
    #    tmp_list = list(filter(lambda x: x[0] not in stop_words, ehr_age[mrn]))
    #    if len(tmp_list) > dpp['min_seq_len']:
    #        ehr_rid.setdefault(mrn,list()).extend(sorted(tmp_list, key=lambda x: x[1]))
    #    else:
    #        count_short += 1
   
    # create new vocabulary
    with open(os.path.join(exp_dir, 'cohort-new-vocab.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['LABEL', 'CODE'])
        cod = 1 ##vocabulary starting from 1 for padding
        new_vidx = {}
        for idx, lab in vidx.items():
            if lab not in stop_words:
                wr.writerow([lab, cod])
                new_vidx[lab] = int(cod)
                cod += 1
 
    if test_set is not None:  
        ehr_rid_ts = {}
        for mrn in pehrs_ts.keys():
            tmp_list = [pehrs_ts[mrn]['events'][idx][0:2]
                        for idx in range(len(pehrs_ts[mrn]['events']))]
            if len(tmp_list) > dpp['min_seq_len']:
                ehr_rid_ts.setdefault(mrn, list()).extend(sorted(tmp_list, 
                                                          key=lambda x: x[1]))
        print("Starting set and shuffle subsequences on test set...")
        ehr_subseq_ts, _, _ = COHORTsubseq_set_shuffle(ehr_rid_ts, f_terms + stop_words)
        print("Ended set and shuffle subsequences on test set. Time: {0} seconds\n".format(round(time() - start, 2)))
        print("Dropped {0} sequences from test set.\n"
              "Initial numerosity: {1} -- Final numerosity: {2}\n".format(len(pehrs_ts) - len(ehr_subseq_ts), 
        len(pehrs_ts), len(ehr_subseq_ts)))
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

    print('Starting set and shuffle subsequences on training set...')
    ehr_subseq, tmp_count_short, tmp_count_long = COHORTsubseq_set_shuffle(ehr_age, stop_words)
    count_short += tmp_count_short
    count_long += tmp_count_long
    print('Ended set and shuffle subsequences on training set. Time: {0} seconds\n'.format(round(time() - start, 2)))

    # Report statistics
    print("Training set statistics:")
    print("Number of dropped sequences: {0} (< {1}),"
          " number of trimmed sequences: {2} (> {3})"
          " Initial total number of patients: {4}"
          " Current number of patients: {5}\n".format(count_short,
                                                   dpp['min_seq_len'],
                                                   count_long,
                                                   dpp['max_seq_len'],
                                                   len(pehrs),
                                                   len(ehr_subseq)))
    
    f_seq_len = [len(ehr_subseq[mrn]) for mrn in ehr_subseq]
    print("The average length of ehr sequences is:"
          " {0:.2f} (Std: {1:.2f})".format(np.mean(f_seq_len),
                                       np.std(f_seq_len)))
    print("Median length of ehr sequences: {0}".format(np.median(f_seq_len)))
    print("The sequence length ranges"
          " from {0} to {1}\n".format(min(f_seq_len),
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
    parser.add_argument('--test_set', dest='test_set', default=None, 
                        help='Add fold')
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
