import torch
import numpy as np

def evaluate(model, loss_fn, data_iterator, metrics, best_eval=False):
    model.eval()
    summ = []
    encoded_list = []
    mrn_list = []

    with torch.no_grad():
        length_vect = []
        for idx, (batch, mrn, lengths) in enumerate(data_iterator):
            batch = batch.cuda()
            out, encoded = model(batch, lengths)
            max_seq_len = max(lengths)
            for i, l in enumerate(lengths):
                mask = torch.FloatTensor([1]*l+[0]*(max_seq_len-l)).cuda()
                out[i] = out[i].clone()*mask
            loss = loss_fn(out, batch)
            out.cpu()
            encoded.cpu()
            summary_batch = {metric:metrics[metric](out, batch).item()
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
               
            if best_eval:
                encoded_list.extend(encoded.tolist())
                mrn_list.extend(mrn)

        length_vect = [len(e) for e in encoded_list]
        max_len = max(length_vect)
        encoded_vect = [e + [0]*(max_len-len(e)) for e in encoded_list]
        #for e in encoded_vect:
            #print(len(e))
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = "--".join("{}: {:05.3f}".format(k,v) for k,v in metrics_mean.items())

        return mrn_list, encoded_vect, metrics_mean
