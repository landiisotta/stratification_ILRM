'''
CODE FOR TRAINING AND EVALUATION. 
Model is evaluated when: loss<0.001 or number of epochs is reached.
Best model is saved in the experiment folder.
'''
import csv
import torch
import utils
from utils import model_pars
import numpy as np
from evaluate import evaluate

def train(model, optimizer, loss_fn, data_iterator):
    model.train()
    encoded_list = []
    loss_batch = []
    mrn_list = []
    for idx, (list_batch, list_mrn) in enumerate(data_iterator):
        for batch, mrn in zip(list_batch, list_mrn):
            batch = batch.cuda()
            optimizer.zero_grad()
            out, encoded_vect = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())
            encoded_list.append(np.mean(encoded_vect.tolist(), axis=0).tolist())
            mrn_list.append(mrn)

    loss_mean = np.mean(loss_batch)

    return mrn_list, encoded_list, loss_mean

def train_and_evaluate(model, data_iterator, loss_fn, optimizer, metrics, experiment_folder):
    num_epochs = model_pars['num_epochs']
    for epoch in range(num_epochs):
        print("Epoch {0} of {1}".format(epoch, num_epochs))
        mrn, encoded, loss_mean = train(model, optimizer, loss_fn, data_iterator)
        print("Mean loss: {0}, epoch {1}".format(loss_mean, epoch))
        is_best = loss_mean < 0.001
        if(is_best or epoch == (num_epochs - 1)):

            with open(experiment_folder + '/TRencoded_vect.csv', 'w') as f:
                wr = csv.writer(f, delimiter=',')
                for e in encoded:
                    wr.writerow(e)

            with open(experiment_folder + '/TRmrns.csv', 'w') as f:
                wr = csv.writer(f, delimiter=',')
                for m in mrn:
                    wr.writerow([m])

            with open(experiment_folder + '/TRmetrics.txt', 'w') as f:
                wr = csv.writer(f, delimiter = '\t')
                wr.writerow(["Mean loss:", loss_mean])  
            
            print("-- Found new best  at epoch {0}".format(epoch))
            utils.save_best_model(model, experiment_folder)
            print("Evaluating the model...")
            mrn, encoded, test_metrics = evaluate(model, loss_fn, data_iterator, metrics, best_eval=True)

            return mrn, encoded, test_metrics

