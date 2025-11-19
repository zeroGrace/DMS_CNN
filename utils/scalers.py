import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_provider.data_args import DatasetArgs

class Scaler():
    def __init__(self) -> None:
        pass
    
    def fit(self, ts_avail:pd.DataFrame):
        pass
    
    def transform(self, data:pd.DataFrame):
        pass
    
    def inverse_transform(self, data_scaled):
        pass


        
# z-score       
class StandardScaler(Scaler):
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, ts_df_avail):
        data_valid = ts_df_avail.values
        
        self.mean = data_valid.mean(axis=0)
        self.std = data_valid.std(axis=0)
        
    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data_scaled, features_type):
        f_dim = -1 if features_type == 'MS' else 0
        return (data_scaled * self.std[f_dim:]) + self.mean[f_dim:]


    
# Min-Max
class MinMaxScaler(Scaler):
    def __init__(self):
        self.min = 0
        self.max = 1

    def fit(self, ts_df_avail):
        data_valid = ts_df_avail.values
        
        self.min = data_valid.min(axis=0)
        self.max = data_valid.max(axis=0)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data_scaled, features_type):
        f_dim = -1 if features_type == 'MS' else 0
        return data_scaled * (self.max[f_dim:] - self.min[f_dim:]) + self.min[f_dim:]



class ScaleOperator():
    def __init__(self) -> None:
        
        self.scaler_keys = ['z_score', 'minmax']
        
        self.scaler_type_dict = {
            'z_score': StandardScaler,
            'minmax': MinMaxScaler,
        }

        self.scaler = None


    def ts_scale(self,
                 ts_df:pd.DataFrame, 
                 scaler_key:str, 
                 train_end:int):

        assert scaler_key in self.scaler_keys

        self.scaler = self.scaler_type_dict[scaler_key]()
        
        self.scaler.fit(ts_df.iloc[0:train_end,:])

        return self.scaler.transform(ts_df)

