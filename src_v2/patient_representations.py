from model.data_loader import EHRdata, ehr_collate
from train import train_and_evaluate
from time import time
from datetime import datetime
from torch.utils.data import DataLoader
import model.net as net
import torch
import utils as ut
import argparse
import sys
import csv
import os

"""
Learn patient representations from the EHRs using an autoencoder of CNNs
"""


def learn_patient_representations(indir, outdir, disease_dt):
    # experiment folder with date and time to save the representations
    exp_dir = os.path.join(outdir, '-'.join(
        [disease_dt,
         datetime.now().strftime('%Y-%m-%d-%H-%M-%S')]))
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
    data = EHRdata(indir, ut.dt_files['ehr'])
    data_generator = DataLoader(data,
                                ut.model_param['batch_size'],
                                shuffle=True,
                                collate_fn=ehr_collate)
    print('Cohort Size: {0} -- Max Sequence Length: {1}'.format(
        len(data), ut.padded_seq_len))

    # define model and optimizer
    model = net.ehrEncoding(vocab_size,
                            ut.padded_seq_len,
                            ut.model_param['embedding_size'],
                            ut.model_param['kernel_size'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ut.model_param['learning_rate'],
                                 weight_decay=ut.model_param['weight_decay'])

    # training and evaluation
    model.cuda()
    loss_fn = net.criterion
    print('Training for {} epochs'.format(ut.model_param['num_epochs']))
    mrn, encoded, metrics_avg = train_and_evaluate(
        model, data_generator, loss_fn, optimizer, net.metrics, exp_dir)

    # save results

    # encoded vectors (representations)
    outfile = os.path.join(exp_dir, 'encoded_vect.csv')
    with open(outfile, 'wb') as f:
        wr = csv.writer(f)
        wr.writerows(encoded)

    # MRNs to keep track of the order
    outfile = os.path.join(exp_dir, 'mrns.csv')
    with open(outfile, 'wb') as f:
        wr = csv.writer(f)
        wr.writerows(mrn)

    # metrics (loss and accuracy)
    outfile = os.path.join(exp_dir, 'metrics.txt')
    with open(outfile, 'wb') as f:
        f.write('Mean loss: %.3f' % metrics_avg['loss'])
        f.write('Accuracy: %.3f' % metrics_avg['accuracy'])


# main function

def _process_args():
    parser = argparse.ArgumentParser(
        description='EHR Patient Stratification: derive patient '
        'representations from longitudinal EHRs')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument(dest='outdir', help='Output directory')
    parser.add_argument(dest='disease_dt', help='Disease dataset name')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()

    start = time()
    learn_patient_representations(args.indir,
                                  args.outdir, args.
                                  disease_dt)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 3))

    print ('Task completed')
