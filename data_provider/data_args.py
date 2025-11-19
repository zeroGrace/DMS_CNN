import numpy as np
# import pandas as pd

class DatasetArgs(object):
    def __init__(self, 
                 data_name:str,
                 # seq length info
                 input_lag:int, 
                 horizon:int, 
                 label_len = None,    # int or str
                 # dataset info
                 split_type:str = 'rate', 
                 train_rate:float = 0.7, 
                 val_rate:float = 0.2,
                 train_start_date = None, 
                 val_start_date = None,
                 test_start_date = None,
                 skip_len:int = 1,
                 # variable(channel/feature) info
                 features_type:str ='S',
                 target:str = 'target',
                 # time info
                 freq:str ='h',
                 # batch info
                 batch_size:int = 32):

        self.data_name = data_name
         
        # seq length info
        self.input_lag = input_lag
        self.horizon = horizon
        
        if label_len == 'i':
            self.label_len = input_lag
        elif label_len == 'h':
            self.label_len = horizon
        else:   
            self.label_len = label_len
            
        self.ts_len = None

        # dataset info
        self.obj_num = None
        self.flag_list = ['train', 'val', 'test']

        self.split_type = split_type
        assert split_type in ['rate','date']
        if self.split_type == 'rate':
            self.train_rate = train_rate
            self.val_rate = val_rate
        else: 
            assert self.split_type == 'date'
            self.start_date_list = [train_start_date, 
                                    val_start_date, 
                                    test_start_date]

        self.border1 = []

        self.skip_len = skip_len
        assert skip_len > 0
        
        # variable(channel/feature) info
        self.features_type = features_type
        assert self.features_type in ['S', 'MS', 'M']

        self.var_dim = None
        self.target = target        

        # time info
        self.freq = freq
        self.freq_list = ['m', 'd', 'w', 'h', '15t', 't', 's']  # 't':min
        assert self.freq in self.freq_list

        if self.freq == 'h':
            self.day_point_n = 24
        elif self.freq == '15t':
            self.day_point_n = 96
        elif self.freq == 't':
            self.day_point_n = 24*60
        elif self.freq == 's':
            self.day_point_n = 24*60*60
        
        
        # batch info
        self.batch_size = batch_size



    
