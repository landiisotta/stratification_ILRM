import csv
import torch
import utils
from utils import model_pars
import numpy as np
from evaluate import evaluate

def train(model, optimizer, loss_fn, data_iterator):
    model.train()
    encoded_list = []
    mrn_list = []
    loss_batch = []
    length_vect = []
    for idx, (batch, mrn, lengths) in enumerate(data_iterator):
        batch = batch.cuda()
        model.hidden = model.init_hidden()    
        optimizer.zero_grad() 
        out, encoded_vect = model(batch, lengths)
        max_seq_len = max(lengths)
        length_vect.extend(lengths)
        for i, l in enumerate(lengths):
                mask = torch.FloatTensor([1]*l+[0]*(max_seq_len-l)).cuda()
                out[i] = out[i].clone()*mask
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())
        
        encoded_list.extend(encoded_vect.tolist())
        mrn_list.extend(mrn)
    
    max_len = max(length_vect)
    encoded = [e + [0.0]*(max_len-len(e)) for e in encoded_list]
    loss_mean = np.mean(loss_batch)
    return mrn_list, encoded, loss_mean

def train_and_evaluate(model, data_iterator, loss_fn, optimizer, metrics, experiment_folder):
    num_epochs = model_pars['num_epochs']
    for epoch in range(num_epochs):
        print("Epoch {0} of {1}".format(epoch, num_epochs))
        mrn, encoded, loss_mean = train(model, optimizer, loss_fn, data_iterator)
        print("Mean loss: {0}, epoch {1}".format(loss_mean, epoch))
        is_best = loss_mean < 0.001
        if(is_best or epoch == (num_epochs - 1)):

            with open(experiment_folder + '/LSTM-TRencoded_vect.csv', 'w') as f:
                wr = csv.writer(f, delimiter=',')
                for e in encoded:
                    wr.writerow(e)

            with open(experiment_folder + '/LSTM-TRmrns.csv', 'w') as f:
                wr = csv.writer(f, delimiter=',')
                for m in mrn:
                    wr.writerow([m])
            
            with open(experiment_folder + '/LSTM-TRmetrics.txt', 'w') as f:
                wr = csv.writer(f, delimiter = '\t')
                wr.writerow(["Mean loss:", loss_mean])  
              
            print("-- Found new best  at epoch {0}".format(epoch))
            print("Evaluating the model...")
            mrn, encoded, test_metrics = evaluate(model, loss_fn, data_iterator, metrics, best_eval=True)

            return mrn, encoded, test_metrics
