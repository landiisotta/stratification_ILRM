import csv
import os
import utils
from utils import ehr_file, mt_to_ix_file, model_pars, L, disease_folder
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
from datetime import datetime

def main():
    f = open("outdir.txt", 'r')
    outdir = f.read().rstrip('\n')
    #create an experiment folder tied to date and time where to save output from the model 
    experiment_folder = os.path.expanduser('~/data1/complex_disorders/experiments/') + disease_folder +\
                    '-'.join(map(str, list(datetime.now().timetuple()[:6])))
    os.makedirs(experiment_folder)
    f = open("experiment_folder.txt", 'w') ##path to the experiment folder is saved in a txt file
    f.write(experiment_folder)
    f.close()
    
    ##pass the size of the vocabulary to the model
    with open(os.path.join(outdir, mt_to_ix_file)) as f:
            rd = csv.reader(f)
            vocab_size = 0
            for r in rd:
                vocab_size+=1

    #set random seed for reproducible experiments
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    ##Import data
    data = myData(outdir, ehr_file)
    data_generator = DataLoader(data, model_pars['batch_size'], shuffle=True, collate_fn=my_collate)
    #define model and optimizer
    print("cohort numerosity:{0} -- max_seq_length:{1}".format(len(data), L))
    model = net.ehrEncoding(vocab_size, L, model_pars['embedding_dim'], model_pars['kernel_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_pars['learning_rate'], weight_decay=1e-5)

    #start the unsupervised training and evaluation
    model.cuda()
    loss_fn = net.criterion
    print("Starting training for {} epochs...".format(model_pars['num_epochs']))
    mrn, encoded, metrics_avg = train_and_evaluate(model, data_generator, loss_fn, optimizer, metrics, experiment_folder)
    
    ##save encoded vectors, medical record number list (to keep track of the order) and metric (loss and accuracy)
    with open(experiment_folder + '/encoded_vect.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for e in encoded:
            wr.writerow(e)

    with open(experiment_folder + '/mrns.csv', 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for m in mrn:
            wr.writerow([m])

    with open(experiment_folder + '/metrics.txt', 'w') as f:
        wr = csv.writer(f, delimiter='\t')
        #for m, v in metrics_average.items():
        #    wr.writerow([m, v])
        wr.writerow(["Mean loss:", metrics_avg['loss']])
        wr.writerow(["Accuracy:", metrics_avg['accuracy']])
 
if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" %(time.time() - start_time))
