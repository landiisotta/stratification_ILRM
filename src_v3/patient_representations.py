from model.data_loader import EHRdata, ehr_collate
from gensim.models import Word2Vec
from train import train_and_evaluate
from time import time
from datetime import datetime
from torch.utils.data import DataLoader
import clustering as clu
import model.net as net
import torch.nn as nn
import torch
import utils as ut
import argparse
import sys
import csv
import os

"""
Learn patient representations from the EHRs using an autoencoder of CNNs
"""


def learn_patient_representations(indir,
                                  outdir,
                                  cv=False,
                                  sampling=None,
                                  emb_filename=None):

    # experiment folder with date and time to save the representations
    exp_dir = os.path.join(outdir, '-'.join(
        [indir,
         datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), '']))
    os.makedirs(exp_dir)

    # get the vocabulary size
    fvocab = os.path.join(indir, ut.dt_files['vocab'])
    with open(fvocab) as f:
        rd = csv.reader(f)
        next(rd)
        vocab = {}
        for r in rd:
            tkn = r[0].split('::')
            tkn[1] = tkn[1].capitalize()
            vocab[int(r[1])] = '::'.join(tkn)
        vocab_size = len(vocab) + 1
    print('Vocabulary size: {0}'.format(vocab_size))

    # load pre-computed embeddings
    if emb_filename is not None:
        model = Word2Vec.load(emb_filename)
        embs = model.wv
        del model
        print('Loaded pre-computed embeddings for {0} concepts'.format(
            len(embs.vocab)))
    else:
        embs = None

    # set random seed for experiment reproducibility
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # load data
    data_tr = EHRdata(indir, ut.dt_files['ehr-tr'], sampling)
    data_generator_tr = DataLoader(data_tr,
        ut.model_param['batch_size'],
        shuffle=True,
        collate_fn=ehr_collate)
    if cv:
        data_ts = EHRdata(indir, ut.dt_files['ehr-ts'], sampling)
        data_generator_ts = DataLoader(data_ts,
        ut.model_param['batch_size'],
        shuffle=True,
        collate_fn=ehr_collate)
    else:
        data_generator_ts = data_generator_tr

    print('Cohort Size: {0} -- Max Sequence Length: {1}\n'.format(
        len(data_tr), ut.len_padded))

    # define model and optimizer
    print('Learning rate: {0}'.format(ut.model_param['learning_rate']))
    print('Batch size: {0}'.format(ut.model_param['batch_size']))
    print('Kernel size: {0}\n'.format(ut.model_param['kernel_size']))

    model = net.ehrEncoding(vocab_size=vocab_size,
                            max_seq_len=ut.len_padded,
                            emb_size=ut.model_param['embedding_size'],
                            kernel_size=ut.model_param['kernel_size'],
                            pre_embs=embs,
                            vocab=vocab
                            )

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ut.model_param['learning_rate'],
                                 weight_decay=ut.model_param['weight_decay'])

    # training and evaluation
    if torch.cuda.device_count() > 1:
        print('No. of GPUs: {0}\n'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        model.cuda()
        print('No. of GPUs: 1\n')

    # model.cuda()
    loss_fn = net.criterion
    print('Training for {} epochs\n'.format(ut.model_param['num_epochs']))
    mrn, encoded, encoded_avg, metrics_avg = train_and_evaluate(model, 
                                                                data_generator_tr, 
                                                                data_generator_ts, 
                                                                loss_fn, 
                                                                optimizer, 
                                                                net.metrics, 
                                                                exp_dir)

    # save results

    # encoded vectors (representations)
    outfile = os.path.join(exp_dir, 'encoded-avg_vect.csv')
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "ENCODED-AVG"])
        for m, e in zip(mrn, encoded_avg):
            wr.writerow([m, e])

    outfile = os.path.join(exp_dir, 'encoded_vect.csv')
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "ENCODED-SUBSEQ"])
        for m, evs in zip(mrn, encoded):
            for e in evs:
                wr.writerow([m] + e)

    # metrics (loss and accuracy)
    outfile = os.path.join(exp_dir, 'metrics.txt')
    with open(outfile, 'w') as f:
        f.write('Mean loss: %.3f\n' % metrics_avg['loss'])
        f.write('Accuracy: %.3f\n' % metrics_avg['accuracy'])
   
    # ehr subseq with age in days
    outfile = os.path.join(exp_dir, 'cohort-ehr-subseq-aid.csv')
    with open(indir, 'cohort-new-ehr-aid.csv') as f:
        rd = csv.reader(f)
        ehr_aid = {}
        for r in rd:
            ehr_aid.setdefault(r[0], list()).append([int(r[1]), int(r[2]]))
    ehr_subseq = {}
    for batch, list_m in data_generator:
        for b, m in zip(batch, list_m):
            if len(b) == 1:
                ehr_subseq[m] = [ehr_aid[m][0][0], ehr_aid[m][-1][0]] \
                                + [e[1] for e in ehr_aid[m]]   
            else:
                seq = [e[1] for e in ehr_aid[m]]
                nseq, nleft = divmod(len(seq), ut.len_padded)
                seq = seq + [0] * \
                     (ut.len_padded - nleft - (ut.seq_overlap - 1) * nseq)    
                for i in range(0, len(seq) - ut.len_padded + 1,
                           ut.len_padded - ut.seq_overlap + 1):
                    ehr_subseq.setdefault(m, list()).append([ehr_aid[m][i][0], 
                                                             ehr_aid[m][i + ut.len_padded][0]] \
                                                             + seq[i:i + ut.len_padded])
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "AID_start", "AID_end", "EHRsubseq"])
        for m, subseq in ehr_aid.items():
            for seq in subseq:
                wr.writerow([m] + [s in seq if s != 0])

    return


# main function

def _process_args():
    parser = argparse.ArgumentParser(
        description='EHR Patient Stratification: derive patient '
        'representations from longitudinal EHRs')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument(dest='outdir', help='Output directory')
    parser.add_argument(dest='disease_dt', help='Disease dataset name')
    parser.add_argument('--eval-baseline', dest='b', action='store_true',
                        help='Evaluate the baseline')
    parser.add_argument('-s', default=None, type=int,
                        help='Enable sub-sampling with data size '
                        '(default: None)')
    parser.add_argument('-e', default=None,
                        help='Pre-computed embeddings (default: None)')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print ('')

    start = time()
    learn_patient_representations(indir=args.indir,
                                  outdir=args.outdir,
                                  disease_dt=args.disease_dt,
                                  eval_baseline=args.b,
                                  sampling=args.s,
                                  emb_filename=args.e)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print ('Task completed\n')