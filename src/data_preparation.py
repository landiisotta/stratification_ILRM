'''
Drop mrns from the previous step and create a new vocabulary [1:len(vocab)] after eliminating the
terms dropped with the patients with too short sequences, if any. Add OTH disease label for the random patients in
cohort-mrn_diseases.csv (MRN, list of disorders strings)
Save new vocabulary (cohort-new_vocab.csv) and new ehrs (cohort-new_ehr.csv).
'''
import csv
import os
import numpy as np

def data_preparation(outdir):
    with open(os.path.join(outdir, 'ehr-shuffle.csv')) as f:
        rd = csv.reader(f)
        ehr_shuffle = {}
        for r in rd:
            ehr_shuffle.setdefault(r[0], list()).extend(r[1::])
    with open(os.path.join(outdir, 'list_mrnToDrop.csv')) as f:
        rd = csv.reader(f)
        mrnToDrop = next(rd)
    with open(os.path.join(outdir, 'stop-words.csv')) as f:
        rd = csv.reader(f)
        stop_words = next(rd)
    with open(os.path.join(outdir, 'cohort-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        ix_to_mt = {}
        for r in rd:
            if r[1] not in stop_words:
                ix_to_mt[r[1]] = r[0]
    with open(os.path.join(outdir, 'cohort-diseases.csv')) as f:
        rd = csv.reader(f)
        code_disease = {}
        for r in rd:
            for c in r[1::]:
                code_disease[c] = r[0]
    with open(os.path.join(outdir, 'cohort-mrns_icds.csv')) as f:
        rd = csv.reader(f)
        mrn_disease = {}
        for r in rd:
            for c in r[1::]:
                mrn_disease.setdefault(r[0], set()).add(code_disease[c])

    ehr_shuffleRid = {}
    for mrn in ehr_shuffle:
        if mrn not in mrnToDrop:
            ehr_shuffleRid[mrn] = ehr_shuffle[mrn]

    ix_term = set()
    for mrn, idx in ehr_shuffleRid.items():
        for i in idx:
            if i not in stop_words:
                ix_term.add(i)

    vocab_trans = {}
    new_vocab = {}
    for i, el in enumerate(ix_term):
        vocab_trans[el] = i+1
        new_vocab[ix_to_mt[el]] = i+1

    new_ehr = {}
    for mrn, seq in ehr_shuffleRid.items():
        new_ehr[mrn] = []        
        for s in seq:
            if s not in stop_words:
                new_ehr[mrn] += [vocab_trans[str(s)]]
 
    for mrn in new_ehr:
        if mrn not in mrn_disease:
            mrn_disease[mrn] = ['OTH']

    with open(os.path.join(outdir, 'cohort-new_vocab.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['LABEL', 'CODE'])
        for l, c in new_vocab.items():
            wr.writerow([l, c])
    with open(os.path.join(outdir, 'cohort-mrn_diseases.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for mrn, dis in mrn_disease.items():
            wr.writerow([mrn] + [d for d in dis])
    with open(os.path.join(outdir, 'cohort-new_ehr.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for mrn, seq in new_ehr.items():
            wr.writerow([mrn] + [s for s in seq])

