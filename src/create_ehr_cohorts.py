"""
In utils.py specify 
1. the name of the folder in the ../data folder
that will contain the cohort files.
2. the parameters needed to create the cohort files, i.e.
the minimum number of diagnosis per patient (data_preprocessing_pars[min_diag]); the number of random
patients to add to the dataset (data_preprocessing_pars[n_rndm]). 
3. medical terms to return (icd9/10, medication, lab, procedure, cpt) (dtype).
4. list of the snomed disease names to select (diseases).

Output:
cohort-vocab.csv: vocabulary of the medical terms from the selected cohort [0; len(vocab)-1]
cohort-ehr.csv: MRN, EVENT (from vocabulary), AGE_IN_DAYS [header yes]
cohort-person.csv: MRN, GENDER, RACE, MOB, YOB [header yes]
cohort-diseases.csv: vocabulary with the selected diseases as keys and the list of 
corresponding icd9/10 codes as items 
cohort-mrns_icds.csv: MRN, LIST OF ICD9/10 DISEASE CODES selected [header no]
"""

from calendar import month_name
from datetime import datetime
import random
import csv
import os

import utils
from utils import data_folder, diseases, dtype, data_preprocessing_pars

def _load_icd_map(filename):
    with open(filename) as f:
        mp = {}
        for l in f:
            tkn = l.split()
            mp.setdefault(tkn[0], set()).add(tkn[1])
        return mp


def _load_dictionaries(d, ctg):
    # lookup (code, code_label)
    flookup = '%s/lookup-%s.csv' % (d, ctg)
    lkp = {}
    with open(flookup, encoding='utf-8') as f:
        rd = csv.reader(f)
        next(rd)
        lkp = {}
        for r in rd:
            label = _validate_code_label(r[1])
            if label is not None:
                lkp[r[0]] = label

    # annotations (code, cui, ontology_id, label)
    fannot = '%s/annotation-%s.csv' % (d, ctg)
    annot = {}
    with open(fannot) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            ant = '::'.join([r[3], r[2], r[1]])
            annot.setdefault(r[0], set()).add(ant)

    return (lkp, annot)


def _validate_code_label(s):
    pnct = '!"#$%&\'*+,-./:;<=>?@\^_`|~ '
    bracket = set(['()', '[]', '{}'])
    record = set(['please', 'not available', 'unclassified',
              'billing', 'profile', 'msdw_unknown',
              'unknown'])
    s = ' '.join(s.split()).strip(pnct)
    for brk in bracket:
        try:
            if s[0] == brk[0] and s[-1] == brk[1]:
                s = s[1:-1]
        except Exception:
            pass

    if s in record:
        return

    try:
        float(s)
        return
    except Exception:
        pass

    if len(s) <= 2:
        return

    return s


def _valid_patient_label(s):
    demog = set(['available', 'unk', 'unknown', 'not available',
             'pt declined', 'msdw_unknown', 'msdw', 'none',
             'other', 'decline'])

    if len(s) < 2 or s in demog:
        return False
    return True


def _format_icd9(cod):
    if cod.startswith('E'):
        brk = 4
    else:
        brk = 3

    if len(cod) == brk:
        return cod

    return cod[:brk] + '.' + cod[brk:]


def _save_dataset(filename, dt):
    with open(filename, 'w') as f:
        wr = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        wr.writerows(dt)

def create_ehr_cohorts():
    """
    Label checker variables
    """

    months = set(m.lower() for m in month_name[1:]) ##month_name[1:] --> list of month names with capital letter


    """
    Functions
    """




    """
    Main script
    """

# parameters
    dt = 'msdw2b'
    datadir = '/home/riccardo/data1/datasets/%s/' % dt
    ehrdir = datadir + 'ehr-csv/'

# snomed diseases to search SEE UTILS
#diseases = []

# minimum number of query diagnosis per patient SEE UTILS
#min_diagn = 3

# number of random patients to integrate the dataset with SEE UTILS
#n_rndm = 2000

# include the patient with specified T2D group
    check_t2d = False

# data types to include in the dataset SEE UTILS
#dtype = ['icd9', 'icd10', 'medication']

# data types to apply the normalizad labels
    to_normalize = set(['cpt', 'medication', 'lab', 'procedure'])

    print('Diseases:', ' - '.join(diseases))

    """
    ##################
    Start to comment from here if you want to specify manually ICD-9/10s
    ##################
    """

# load data
    filename = datadir + 'mappings/icd9-to-snomed.csv'
    with open(filename, encoding='utf-8') as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd)
        mp_snomed = {}
        for r in rd:
            if r[2] == '1':
                mp_snomed.setdefault(r[8].lower(), set()).add(r[0])
##mp_snomed --> dictionary 'snomed_fsn'=icd9_code

##9to10/10to9 dictionaries (no points in codes)
    icd9to10 = _load_icd_map(datadir + 'mappings/icd9-to-icd10.txt')
    icd10to9 = _load_icd_map(datadir + 'mappings/icd10-to-icd9.txt')


# get query ICD codes
##icds = set of icd9/10 codes related to diseases
    icds = set()
    disease_icds_dict = {}
    for q in sorted(diseases):
        qres = filter(lambda x: x.find(q)!=-1, mp_snomed)
        icd_codes = []
        for r in qres:
            icds |= mp_snomed[r]
            icd_codes.extend(list(mp_snomed[r]))
            for d in mp_snomed[r]:
                try:
                    for c in icd9to10[d.replace('.', '')]:
                        icds.add(c[:3] + '.' + c[3:])
                        icd_codes.append(c[:3] + '.' + c[3:])
                except Exception:
                    pass
        disease_icds_dict[q] = list(set(icd_codes))

    """
    ##################
    End the comment here if you want to specify manually ICD-9/10s
    Add the command: icds = set([COD1, COD2, ..., CODn])
    ##################
    """


# get MRNs from ICD-9/10 files
    filenames = [ehrdir + 'icd9/person-icd9.csv',
                ehrdir + 'icd10/person-icd10.csv']
    icd_mrns = {}
    mrns_icds = {}
    all_mrns = {}
    for fname in filenames:
        with open(fname) as f:
            rd = csv.reader(f)
            for r in rd:
                if r[1] in icds:
                    icd_mrns.setdefault(r[0], set()).add((r[1], r[2]))
                    mrns_icds.setdefault(r[0], set()).add(r[1])
                all_mrns.setdefault(r[0], set()).add(r[1])
##icd_mrns=dictionary with 'mrn'={(icd9/10_diseasesDiag, time_of_diag)}
##all_mrns=dictionary with all 'mrn'={icd9/10_diag}

##mrns=set of mrns of patients with at leat min_diag diagnosis of diseases
##oths=set of all mrns with at least min_diag general diagnosis
    mrns = set(m for m in icd_mrns if len(icd_mrns[m]) >= data_preprocessing_pars['min_diagn'])
    oths = set(m for m in all_mrns if len(all_mrns[m]) >= data_preprocessing_pars['min_diagn'])

##remove the mrns that have a diseases diagnosis from the set of all patients
##shuffle the list and select n_rndm mrns, then add those to mrns set
    rndm = list(oths - (oths & mrns))
    random.shuffle(rndm)
    if data_preprocessing_pars['n_rndm'] > 0 and len(rndm) > data_preprocessing_pars['n_rndm']:
        rndm = rndm[:data_preprocessing_pars['n_rndm']]
        mrns |= set(rndm)
    print('No of Patients:', len(mrns))


# check T2D patients
    if check_t2d:
        filename = datadir + 't2d-groups/mrn-t2d-groups.csv'
        with open(filename) as f:
            rd = csv.reader(f)
            next(rd)
            t2d_mrns = set(r[0] for r in rd)
        mrns |= t2d_mrns


# create the dataset
    vocab = set()
    dt = set()
    for ctg in dtype:
        print('Processing:', ctg)
        if ctg == 'icd10':
            lookup, annot = _load_dictionaries(ehrdir + 'icd9', 'icd9')
            filename = '%s%s/person-%s.csv' % (ehrdir, ctg, ctg)
        else:
            d = ehrdir + ctg
            lookup, annot = _load_dictionaries(d, ctg)
            filename = '%s/person-%s.csv' % (d, ctg)
        with open(filename) as f:
            rd = csv.reader(f)
            for i, r in enumerate(rd):
                if r[0] not in mrns:
                    continue

                try:
                    age_in_days = int(r[2])
                except Exception:
                    continue

                cods = None

            # map ICD10 to ICD9
                if ctg == 'icd10':
                    try:
                        icd9s = icd10to9[r[1].replace('.', '')]
                        cods = [set(_format_icd9(c) for c in icd9s)]
                    except Exception:
                        continue

                if cods is None:
                    cods = [r[1]]

                lbl = set()
                for c in cods:
                # format the label
                    if ctg.startswith('icd'):
                        try:
                            lbl |= set(['::'.join([ctg, a]) for a in annot[c]])
                        except Exception:
                            continue

                    elif c not in lookup:
                        continue

                    elif ctg in to_normalize and c in annot:
                        lbl |= set(['::'.join([ctg, a]) for a in annot[c]])

                    else:
                        lbl |= set(['::'.join([ctg, lookup[c]])])

                for l in lbl:
                    dt.add((r[0], l, age_in_days))
                    vocab.add(l)

##dt = set of (mrn, dtype_labels_codes, time) (not unique mrns)
##vocab = set of dtype_labels_codes (unique)

# load patient details
##dictionary of person details (only diseases selected)
    filename = ehrdir + 'person/person-detail.csv'
    person = {}
    with open(filename) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            r = list(map(str.lower, map(str.strip, r)))
            if r[0] not in mrns:
                continue
            prs = person.setdefault(r[0], ['' for _ in range(5)])

            if len(prs[0]) == 0:
                if r[1] == 'male' or r[1] == 'female':
                    prs[0] = r[1]

            if len(prs[1]) == 0:
                if _valid_patient_label(r[2]):
                    if r[2] == 'atino':
                        prs[1] = 'hispanic/latino'
                    else:
                        prs[1] = r[2]

            if len(prs[2]) == 0 and r[3] in months:
                prs[2] = r[3]

            if len(prs[3]) == 0:
                try:
                    int(r[4])
                    prs[3] = r[4]
                except Exception:
                    pass

            person[r[0]] = prs


# save the dataset
    outdir = data_folder + 'cohorts/' + \
        '-'.join(map(str, list(datetime.now().timetuple()[:6])))
    os.makedirs(outdir)

    ivcb = {p: i for i, p in enumerate(sorted(vocab))}
    out_vcb = [('LABEL', 'CODE')] + sorted(ivcb.items())
    outfile = outdir + '/cohort-vocab.csv'
    _save_dataset(outfile, out_vcb)

    out_dt = [('MRN', 'EVENT', 'AGE_IN_DAYS')]
    for r in sorted(dt):
        out_dt.append((r[0], ivcb[r[1]], r[2]))
    outfile = outdir + '/cohort-ehr.csv'
    _save_dataset(outfile, out_dt)

    outfile = outdir + '/cohort-person.csv'
    out_person = [('MRN', 'GENDER', 'RACE', 'MOB', 'YOB')] + \
        [([el[0]] + el[1]) for el in sorted(person.items())]
    _save_dataset(outfile, out_person)

    out_dis = []
    for dis, icds in disease_icds_dict.items():
        out_dis.append([dis] + icds)
    outfile = outdir + '/cohort-diseases.csv'
    _save_dataset(outfile, out_dis)

    out_mrns = []
    for mrn, icds in mrns_icds.items():
        out_mrns.append([mrn] + list(icds))
    outfile = outdir + '/cohort-mrns_icds.csv'
    _save_dataset(outfile, out_mrns)

    return outdir
