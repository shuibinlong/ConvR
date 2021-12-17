import torch
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_for_tail(eval_data, model, device, data, descending):
    # TODO: add reverse relations as https://github.com/TimDettmers/ConvE/ in training and evaluation
    hits = []
    ranks = []
    ent_rel_multi_t = data['entity_relation']['as_tail']
    for _ in range(10):  # need at most Hits@10
        hits.append([])

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r = batch_data[2].to(device)
        _, pred = model(eval_h, eval_r)  # evaluate corruptions by replacing the object, i.e. tail entity

        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_h.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_t = ent_rel_multi_t[eval_h[i].item()][eval_r[i].item()]

            pred_value = pred[i][eval_t[i].item()].item()
            pred[i][filter_t] = 0.0
            pred[i][eval_t[i].item()] = pred_value

        _, index = torch.sort(pred, 1, descending=True)  # pred: (batch_size, ent_count)
        index = index.cpu().numpy()  # index: (batch_size)

        for i in range(eval_h.size(0)):
            # find the rank of the target entities
            rank = np.where(index[i] == eval_t[i].item())[0][0]

            # rank+1, since the rank starts with 1 not 0
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    return hits, ranks

def output_eval_tail(results, data_name):
    hits = np.array(results[0])
    ranks = np.array(results[1])
    r_ranks = 1.0 / ranks  # compute reciprocal rank

    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank)
    print('For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits[9].mean(), hits[2].mean(), hits[0].mean()))
    print('For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks.mean(), r_ranks.mean()))