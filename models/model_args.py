import numpy as np
import torch.nn as nn

class ModelArgs(object):
    def __init__(self, 
                 batch_size:int,
                 epochs:int,
                 lr:float,
                 ):
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr



class Args_DMSCNN(ModelArgs):
    def __init__(self, 
                # self
                sight_list:list,
                conv_ks_1:int,
                decomp_ks_list:list,
                trend_hidden_layers:list = [128,],
                dropout: float = 0.0,
                # super
                batch_size: int = 32, 
                epochs: int = 50, 
                lr: float = 0.001, 
            ):
        super().__init__(batch_size, epochs, lr)

        self.sight_list = sight_list
        self.conv_ks_1 = conv_ks_1
        self.decomp_ks_list = decomp_ks_list
        self.trend_hidden_layers = trend_hidden_layers
        self.dropout = dropout
    

