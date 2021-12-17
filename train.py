import torch
import logging 
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_without_label(data, model, optimizer, device):
    full_loss = []
    model.train()
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        optimizer.zero_grad()
        loss, _ = model(h, r, t)
        # loss = loss.mean()
        loss.backward()
        optimizer.step()
        full_loss.append(loss.item())
    return full_loss