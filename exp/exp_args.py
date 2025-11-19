import numpy as np
# import pandas as pd
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class ExpArgs(object):
    def __init__(self,
                 repeat_time,
                 data_name:str,
                 # path
                 root_path:str,
                 # device  
                 cuda_id:int = None,
                 # time label
                 timeenc:int = 0,
                 # model setting
                 forecast_mode:str = 'l',
                 model_name:str = 'MLP',
                 patience:int = 7,
                 lradj:bool = True,
                 adj_type:str = 'type1',
                 train_only:bool = False,
                 # series selection
                 select_series:bool = False,
                 select_id_list:list = None,
                 # scale
                 is_scale:bool = True,
                 scale_type:str = 'minmax'
                 ):
        
        
        # dir for Not Code
        dir_nc = '_NC'

        self.repeat_time = repeat_time
        
        # device
        if cuda_id == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
        
        self.data_name = data_name
        
        # path
        self.root_path = root_path
        self.data_dir =  os.path.join(root_path, dir_nc, 'data', self.data_name) 

        self.ds_path = os.path.join(root_path, dir_nc, 'data_ds')
        self.checkpoints = os.path.join(root_path, dir_nc, 'checkpoints')
        self.result_path = os.path.join(root_path, dir_nc, 'results')
        self.fig_path = os.path.join(root_path, dir_nc, 'figs')
        
        # time label
        self.timeenc = timeenc
        # assert self.timeenc in [0,1]

        # model setting
        # l: local; g: global
        self.forecast_mode = forecast_mode
        assert self.forecast_mode in ['l', 'g'] 
        self.model_name = model_name
        self.patience = patience
        self.lradj = lradj
        self.adj_type = adj_type
        self.train_only = train_only
        
        # series selection
        self.select_series = select_series
        
        if self.select_series == True:
            self.select_id_list = select_id_list
            assert len(select_id_list) >= 1

        self.ts_id_list = []

        # scale
        self.is_scale = is_scale
        
        if self.is_scale == True:
            self.scale_type = scale_type
            assert self.scale_type in ['z_score', 'minmax', 'boxcox']
        else:
            self.scale_type = None

        self.scaler_dict = {}


        
        
        
        
        
