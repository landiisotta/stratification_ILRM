import csv
import os
import utils
from utils import ehr_file, mt_to_ix_file, model_pars
from model.data_loader import myData, my_collate
import model.net as net
from model.net import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import train
from train import train_and_evaluate 
import evaluate
from evaluate import evaluate

def main():
    f = open("outdir.txt", 'r')
    outdir = f.read().rstrip('\n')
    f = open("experiment_folder_bin2.txt", 'r')
    experiment_folder = f.read().rstrip('\n')
   
    ##pass the size of the vocabulary to the model
    with open(os.path.join(outdir, mt_to_ix_file)) as f:
            rd = csv.reader(f)
            vocab_size = 0
            for r in rd:
                vocab_size+=1

    #set random seed for reproducible experiments
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)

    ##Import data
    data = myData(outdir, ehr_file)
    data_generator = DataLoader(data, model_pars['batch_size'], shuffle=True, collate_fn=my_collate, drop_last=True)
    #define model and optimizer
    print("cohort numerosity:{0}".format(len(data)))
    model = net.LSTMehrEncoding(vocab_size, model_pars['embedding_dim'], model_pars['batch_size'])
    #model = nn.DataParallel(model, device_ids=[1,2,3])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_pars['learning_rate'], weight_decay=1e-5)

    #start the unsupervised training and evaluation
    model.cuda()
    loss_fn = net.criterion
    print("Starting training for {} epochs...".format(model_pars['num_epochs']))
    mrn, encoded, metrics_avg = train_and_evaluate(model, data_generator, loss_fn, optimizer, metrics, experiment_folder)
    with open(experiment_folder + '/LSTMencoded_vect.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for e in encoded:
            wr.writerow(e)

    with open(experiment_folder + '/LSTMmrns.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for m in mrn:
            wr.writerow([m])

    with open(experiment_folder + '/LSTMmetrics.txt', 'w') as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerow(["Mean loss:", metrics_avg['loss']])
        wr.writerow(["Accuracy:", metrics_avg['accuracy']])
 
if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" %(time.time() - start_time))
