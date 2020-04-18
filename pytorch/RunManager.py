from itertools import product

from collections import namedtuple
from collections import OrderedDict

import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time 
import json
from IPython.display import clear_output
import pandas as pd


    
class RunManager():
    def __init__(self):
        self.epoch_count = 0 
        self.epoch_loss = 0 
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count= 0
        self.run_data = []
        self.run_start_time= None
        
        self.network = None
        self.loader = None
        self.tb =None
        
    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count+=1
        
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        
        self.tb.add_image('images', grid)
        self.tb.add_graph(network, images)
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count = 0 
        self.epoch_loss = 0 
        self.epoch_num_correct = 0
        
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['run_duration'] = run_duration
        results['epoch_duration'] = epoch_duration
        
        for k,v in self.run_params._asdict().items(): results[k] = v
            
        self.run_data.append(results)
        df= pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        clear_output(wait = True)
                
    def track_loss(self,loss):
        self.epoch_loss += loss.item() * self.loader.batch_size
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_number_correct(preds, labels)
        
    @torch.no_grad()   
    def _get_number_correct(self, preds, labels):
        return preds.argmax(dim =1).eq(labels).sum().item()
    
    def save(self, filename):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{filename}.csv')
        
        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
            

            
            