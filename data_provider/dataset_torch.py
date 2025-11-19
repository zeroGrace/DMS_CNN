import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_provider.data_args import DatasetArgs

class DatasetDL(Dataset):
    def __init__(self, 
                 ds:np.ndarray, 
                 d_args:DatasetArgs):
        '''
        Args:
            ds: (sample_num, L, C); 
        '''
        self.input_lag = d_args.input_lag
        self.horizon = d_args.horizon
           
        self.data = torch.from_numpy(ds)


    def __getitem__(self, index: int):
        '''
        Args:
            index: the index of item
        Returns:
            input(lagged value) and output of ith sample in data
        '''
        
        # data:(sample_num, L, C)
        seq_x = self.data[index, 0:self.input_lag, :]
        seq_y = self.data[index, -self.horizon:,:]
        return seq_x, seq_y
    

    def __len__(self):
        return self.data.shape[0]



# with timestamp info
class DatasetWT(Dataset):  
    def __init__(self, 
                 ds:np.ndarray, 
                 stamp:np.ndarray,
                 d_args:DatasetArgs):
        '''
        Args:
            ds: (sample_num, L, C);
            stamp:(sample_num, L, label_num); 
        '''
        self.input_lag = d_args.input_lag
        self.horizon = d_args.horizon
        self.time_stamp = torch.from_numpy(stamp)
        self.data = torch.from_numpy(ds)


    def __getitem__(self, index: int):
        '''
        Args:
            index: the index of item
        '''
        # data:(sample_num, L, C)
        seq_x = self.data[index, 0:self.input_lag, :]
        seq_y = self.data[index, -self.horizon:,:]
        seq_x_mark = self.time_stamp[index, 0:self.input_lag, :]
        seq_y_mark = self.time_stamp[index, -self.horizon:,:]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    

    def __len__(self):
        return self.data.shape[0]

    