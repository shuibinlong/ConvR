import os
import random
import logging
import numpy as np
from tqdm import tqdm

from utils import load_triples, load_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dataset:
    def __init__(self, dataset):
        self.path = os.getcwd()
        self.name = dataset
        self.data = {
            'train': self.read_train(),
            'valid': self.read_valid(),
            'test': self.read_test(),
            'entity': self.read_entity(),
            'relation': self.read_relation(),
            'entity_relation': {}
        }

        self.gen_entity_relation_multidata()
    
    def read_train(self):
        logging.info(' Loading training data '.center(100, '-'))
        return load_triples(os.path.join(self.path, 'data', self.name, 'train2id.txt'))
    
    def read_valid(self):
        logging.info(' Loading validation data '.center(100, '-'))
        return load_triples(os.path.join(self.path, 'data', self.name, 'valid2id.txt'))

    def read_test(self):
        logging.info(' Loading testing data '.center(100, '-'))
        return load_triples(os.path.join(self.path, 'data', self.name, 'test2id.txt'))
    
    def read_entity(self):
        logging.info(' Loading entity id '.center(100, '-'))
        return load_ids(os.path.join(self.path, 'data', self.name, 'entity2id.txt'))
    
    def read_relation(self):
        logging.info(' Loading realtion id '.center(100, '-'))
        return load_ids(os.path.join(self.path, 'data', self.name, 'relation2id.txt'))
    
    def gen_entity_relation_multidata(self):
        logging.info(' Generating entity-relation dictionaries to accelerate evaluation process '.center(100, '-'))
        full_data = self.data['train'] + self.data['valid'] + self.data['test']
        self.data['entity_relation']['as_head'] = {}
        self.data['entity_relation']['as_tail'] = {}
        for i in self.data['entity']:
            self.data['entity_relation']['as_head'][i] = {}
            self.data['entity_relation']['as_tail'][i] = {}
            for j in self.data['relation']:
                self.data['entity_relation']['as_head'][i][j] = []
                self.data['entity_relation']['as_tail'][i][j] = []
        for triple in full_data:
            h, t, r = triple
            self.data['entity_relation']['as_head'][t][r].append(h)
            self.data['entity_relation']['as_tail'][h][r].append(t)