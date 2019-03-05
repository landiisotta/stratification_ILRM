from model.data_loader import EHRdata, ehr_collate
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
                                  disease_dt,
                                  eval_baseline=False,
                                  sampling=None):

    # experiment folder with date and time to save the representations
    exp_dir = os.path.join(outdir, '-'.join(
        [disease_dt,
         datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), 'nobn-noact-norelu-10']))
    os.makedirs(exp_dir)

    # get the vocabulary size
    fvocab = os.path.join(indir, ut.dt_files['vocab'])
    with open(fvocab) as f:
        rd = csv.reader(f)
        next(rd)
        vocab_size = sum(1 for r in rd) + 1
    print('Vocabulary size: {0}'.format(vocab_size))

    # set random seed for experiment reproducibility
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    # load data
    data = EHRdata(indir, ut.dt_files['ehr'], sampling)
    data_generator = DataLoader(
        data,
        ut.model_param['batch_size'],
        shuffle=True,
        collate_fn=ehr_collate)

    print('Cohort Size: {0} -- Max Sequence Length: {1}\n'.format(
        len(data), ut.len_padded))

    # define model and optimizer
    print('Learning rate: {0}'.format(ut.model_param['learning_rate']))
    print('Batch size: {0}'.format(ut.model_param['batch_size']))
    print('Kernel size: {0}\n'.format(ut.model_param['kernel_size']))

    model = net.ehrEncoding(vocab_size,
                            ut.len_padded,
                            ut.model_param['embedding_size'],
                            ut.model_param['kernel_size'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ut.model_param['learning_rate'],
                                 weight_decay=ut.model_param['weight_decay'])

    # training and evaluation
    if torch.cuda.device_count() > 1:
        print('No. of GPUs: {0}\n'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        print('No. of GPUs: 1\n')

    # model.cuda()
    loss_fn = net.criterion
    print('Training for {} epochs\n'.format(ut.model_param['num_epochs']))
    mrn, encoded, metrics_avg = train_and_evaluate(
        model, data_generator, loss_fn, optimizer, net.metrics, exp_dir)

    # save results

    # encoded vectors (representations)
    outfile = os.path.join(exp_dir, 'encoded_vect.csv')
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(encoded)

    # MRNs to keep track of the order
    outfile = os.path.join(exp_dir, 'mrns.csv')
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        for m in mrn:
            wr.writerow([m])

    # metrics (loss and accuracy)
    outfile = os.path.join(exp_dir, 'metrics.txt')
    with open(outfile, 'w') as f:
        f.write('Mean loss: %.3f\n' % metrics_avg['loss'])
        f.write('Accuracy: %.3f\n' % metrics_avg['accuracy'])

    # evaluate clustering
    gt_file = os.path.join(indir, ut.dt_files['diseases'])
    gt_disease = clu.load_mrn_disease(gt_file)
    min_clu = 2
    max_clu = 10

    if eval_baseline:
        print('\nRunning clustering on the TF-IDF vectors')
        datafile = os.path.join(indir, ut.dt_files['ehr'])
        mrn_idx, svd_mtx = clu.svd_tfidf(datafile, vocab_size)
        gt_disease_raw = [gt_disease[m][0] for m in mrn_idx]
        clu.eval_hierarchical_clustering(
            svd_mtx, gt_disease_raw, min_clu, max_clu)

    print('\nRunning clustering on the encoded vectors')
    gt_disease_enc = [gt_disease[m][0] for m in mrn]
    clu.eval_hierarchical_clustering(
        encoded, gt_disease_enc, min_clu, max_clu, preproc=True)

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
                        '(defaut: None)')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print ('')

    start = time()
    learn_patient_representations(indir=args.indir,
                                  outdir=args.outdir,
                                  disease_dt=args.disease_dt,
                                  eval_baseline=args.b,
                                  sampling=args.s)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print ('Task completed\n')
