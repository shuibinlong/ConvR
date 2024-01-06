import torch
import logging 
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_without_label(config, data, model, optimizer, device):
    full_loss = []
    rel_cnt = config.get('relation_cnt')
    model.train()
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        # 正关系
        optimizer.zero_grad()
        loss, _ = model(h, r, t)
        # loss = loss.mean()
        loss.backward()
        optimizer.step()

        # 逆关系
        optimizer.zero_grad()
        loss_rev, _ = model(t, r + rel_cnt, h)
        # loss = loss.mean()
        loss_rev.backward()
        optimizer.step()
        full_loss.append((loss.item()+loss_rev.item())/2)
    return full_loss