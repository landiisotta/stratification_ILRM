"""
Model training and evaluation.

The model is evaluated when (1) loss < 0.001 or (2) the number of
epochs is reached. The best model is saved in the experiment folder.
"""

from evaluate import evaluate
from time import time
import utils as ut
import numpy as np
import csv
import os


def train(model, optimizer, loss_fn, data_iter):
    model.train()
    encoded_list = []
    loss_list = []
    mrn_list = []
    for idx, (list_batch, list_mrn) in enumerate(data_iter):
        loss_batch = []

        for batch, mrn in zip(list_batch, list_mrn):
            batch = batch.cuda()
            optimizer.zero_grad()
            out, encoded_vect = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())
            encoded_list.append(
                np.mean(encoded_vect.tolist(), axis=0).tolist())
            mrn_list.append(mrn)

        loss_list.append(np.mean(loss_batch))

    loss_mean = np.mean(loss_list)

    return mrn_list, encoded_list, loss_mean


def train_and_evaluate(model, data_iter, loss_fn,
                       optimizer, metrics, exp_dir):
    loss_vect = []
    n_epoch = ut.model_param['num_epochs']
    for epoch in range(1, n_epoch + 1):
        print('Epoch {0} of {1}'.format(epoch, n_epoch))

        start = time()
        mrn, encoded, loss_mean = train(
            model, optimizer, loss_fn, data_iter)
        print ('-- time = ', round(time() - start, 3))
        print ('-- mean loss: {0}'.format(round(loss_mean, 3)))
        loss_vect.append(loss_mean)

        is_best_1 = loss_mean < 0.01
        is_best_2 = epoch == n_epoch
        if is_best_1 or is_best_2:

            outfile = os.path.join(exp_dir, 'TRencoded_vect.csv')
            with open(outfile, 'w') as f:
                wr = csv.writer(f)
                wr.writerows(encoded)

            outfile = os.path.join(exp_dir, 'TRmrns.csv')
            with open(outfile, 'w') as f:
                wr = csv.writer(f)
                for m in mrn:
                    wr.writerows([m])

            outfile = os.path.join(exp_dir, 'TRmetrics.txt')
            with open(outfile, 'w') as f:
                f.write('Mean Loss: %.3f\n' % loss_mean)

            os.path.join(exp_dir, 'TRlosses.csv')
            with open(outfile, 'wb') as f:
                wr = csv.writer(f)
                wr.writerow(['Epoch', 'Loss'])
                for idx, l in enumerate(loss_vect):
                    wr.writerow([idx, l])

            print('-- Found new best model at epoch {0}'.format(epoch))
            ut.save_best_model(model, exp_dir)

            print('Evaluating the model')
            mrn, encoded, test_metrics = evaluate(
                model, loss_fn, data_iter, metrics, best_eval=True)

            return mrn, encoded, test_metrics
