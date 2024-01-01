import torch
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_for_tail(config, eval_data, model, device, data, descending):
    # TODO: add reverse relations as https://github.com/TimDettmers/ConvE/ in training and evaluation
    hits_tail = []
    hits_head = []
    ranks = []
    ent_rel_multi_t = data['entity_relation']['as_tail']
    ent_rel_multi_h = data['entity_relation']['as_head']
    rel_cnt = config.get('relation_cnt')
    for _ in range(10):  # need at most Hits@10
        hits_tail.append([])
        hits_head.append([])

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r = batch_data[2].to(device)
        _, pred_tail = model(eval_h, eval_r)  # evaluate corruptions by replacing the object, i.e. tail entity
        _, pred_head = model(eval_t, eval_r + rel_cnt)
        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_h.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_t = ent_rel_multi_t[eval_h[i].item()][eval_r[i].item()]
            filter_h = ent_rel_multi_h[eval_t[i].item()][eval_r[i].item()]

            pred_tail_value = pred_tail[i][eval_t[i].item()].item()
            pred_head_value = pred_head[i][eval_h[i].item()].item()

            pred_tail[i][filter_t] = 0.0
            pred_head[i][filter_h] = 0.0

            pred_tail[i][eval_t[i].item()] = pred_tail_value
            pred_head[i][eval_h[i].item()] = pred_head_value

        _, index_tail = torch.sort(pred_tail, 1, descending=True)
        _, index_head = torch.sort(pred_head, 1, descending=True)  # pred: (batch_size, ent_count)

        index_tail = index_tail.cpu().numpy()  # index: (batch_size)
        index_head = index_head.cpu().numpy()

        for i in range(eval_h.size(0)):
            # find the rank of the target entities
            rank_tail = np.where(index_tail[i] == eval_t[i].item())[0][0]
            rank_head = np.where(index_head[i] == eval_h[i].item())[0][0]

            # rank+1, since the rank starts with 1 not 0
            ranks.append(rank_tail + 1)
            ranks.append(rank_head + 1)

            for hits_level in range(10):
                if rank_tail <= hits_level:
                    hits_tail[hits_level].append(1.0)
                else:
                    hits_tail[hits_level].append(0.0)

                if rank_head <= hits_level:
                    hits_head[hits_level].append(1.0)
                else:
                    hits_head[hits_level].append(0.0)

    return hits_tail, hits_head, ranks

def output_eval_tail(results, data_name):
    hits_tail = np.array(results[0])
    hits_head = np.array(results[1])
    ranks = np.array(results[2])
    r_ranks = 1.0 / ranks  # compute reciprocal rank

    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank)
    print('For %s data: hits_tail@10=%.4f - hits_tail@3=%.4f - hits_tail@1=%.4f' % (data_name, hits_tail[9].mean(), hits_tail[2].mean(), hits_tail[0].mean()))
    print('For %s data: hits_head@10=%.4f - hits_head@3=%.4f - hits_head@1=%.4f' % (data_name, hits_head[9].mean(), hits_head[2].mean(), hits_head[0].mean()))
    print('For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks.mean(), r_ranks.mean()))