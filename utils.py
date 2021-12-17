import os
import json
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_config(config_path):
    logging.info(' Loading configuration '.center(100, '-'))
    if not os.path.exists(config_path):
        logging.warning(f'File {config_path} does not exist, empty list is returned.')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return config

def load_triples(file_path):
    tuples = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d triples loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip().split(' ')
                tuples.append(tuple(map(int, record)))
    return tuples

def load_ids(file_path):
    ids = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d of entities/relations loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip()
                try:
                    id = record.split('\t')[1]
                except IndexError:
                    id = record.split(' ')[-1]
                ids.append(int(id))
    return ids