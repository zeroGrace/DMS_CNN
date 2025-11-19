import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from data_provider.data_args import DatasetArgs
from exp.exp_args import ExpArgs
from utils.tools import PrintWithLogging as PWL
from utils.scalers import ScaleOperator
from utils.timeencFeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class DsPre(object):
    def __init__(self, 
                 d_args:DatasetArgs,
                 e_args:ExpArgs): 

        self.d_args = d_args
        self.e_args = e_args

        self.split_type = d_args.split_type
        self.features_type = d_args.features_type
        self.target = d_args.target
        self.scale = e_args.is_scale
        self.timeenc = e_args.timeenc  
        self.freq = d_args.freq
        self.train_only = e_args.train_only

        self.root_path = e_args.root_path
        self.data_path = e_args.data_dir

        self.multi_ds_dict, self.multi_stamp_dict = self.gen_ds()


    def gen_ds(self) -> dict:
        pwl = PWL()
        obj_ds_dict, obj_stamp_dict = self._get_ds()
        
        if self.e_args.forecast_mode == 'l':
            multi_ds_dict = obj_ds_dict
            multi_stamp_dict = obj_stamp_dict
            
            id = self.e_args.ts_id_list[0]
            for flag in self.d_args.flag_list:
                pwl('{} (local): ds {} (sample_n, L, C), stamp {} (sample_n, L, label_num)'.format(flag,
                                                                                                   multi_ds_dict[id][flag].shape,
                                                                                                   multi_stamp_dict[id][flag].shape))
        else:
            ds_dict = {}
            stamp_dict = {}
            for flag in self.d_args.flag_list:
                ds_all_list = [obj_ds_dict[id][flag] for id in self.e_args.ts_id_list]
                stamp_all_list = [obj_stamp_dict[id][flag] for id in self.e_args.ts_id_list]

                ds_dict[flag] = np.concatenate(ds_all_list,axis=0)
                stamp_dict[flag] = np.concatenate(stamp_all_list, axis=0)

                pwl('{} (global): ds {} (sample_n, L, C), stamp {} (sample_n, L, label_num)'.format(flag,
                                                                                                    ds_dict[flag].shape,
                                                                                                    stamp_dict[flag].shape))
                
            multi_ds_dict = {'g': ds_dict}
            multi_stamp_dict = {'g': stamp_dict}
        
        return multi_ds_dict, multi_stamp_dict
    

    def _get_ds(self) -> dict:
        pwl = PWL()

        if self.e_args.select_series == True:
            self.e_args.ts_id_list = self.e_args.select_id_list
            pwl('select series: id {}'.format(self.e_args.ts_id_list))
        else: 
            self.e_args.ts_id_list = list(range(len(os.listdir(self.data_path))))
            pwl('select series: all series')

        self.d_args.obj_num = len(self.e_args.ts_id_list)
        pwl('obj_num: {}'.format(self.d_args.obj_num))

        obj_ds_dict = {}
        obj_stamp_dict = {}
        
        for obj_id in self.e_args.ts_id_list:
            file_name = '{}.csv'.format(str(obj_id))
            df_raw = pd.read_csv(os.path.join(self.data_path, file_name))
            '''
            df shape: (series_L, C+1)
            df_raw.columns: ['datetime', ...(other features), 'target']
            '''

            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('datetime')
            df_raw = df_raw[['datetime'] + cols + [self.target]]

            ts_len = len(df_raw)
            self.d_args.ts_len = ts_len
            
            ds_split = DsSplit(d_args=self.d_args)
            if self.split_type == 'rate':
                ds_split.split_by_rate()
            elif self.split_type == 'date':
                ds_split.split_by_date(ts_start_date = df_raw[['datetime']].iloc[0,0])
            
            self.d_args.border1 = ds_split.border1

            if self.features_type == 'S':
                df_data = df_raw[[self.target]]
            else:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            
            self.d_args.var_dim = df_data.shape[1]

            if self.e_args.is_scale:
                data, scaler = self._data_scale(df_data,
                                                train_end=ds_split.border2[0])
                self.e_args.scaler_dict[obj_id] = scaler
            else:
                data = df_data.values
            
            data_stamp = self._time_label(df_stamp = df_raw[['datetime']])
            self.label_num = data_stamp.shape[1]

            '''
            ds:(sample_num, L, C), ds_stamp:(sample_num, L, label_num)
            '''
            ds_dict, ds_stamp_dict = ds_split.generate_ds(data, data_stamp)
            
            obj_ds_dict[obj_id] = ds_dict
            obj_stamp_dict[obj_id] = ds_stamp_dict
        
        
        pwl('ts_len: {}'.format(self.d_args.ts_len))
        if self.e_args.is_scale:
            pwl('scale_type:{}'.format(self.e_args.scale_type))
        else:
            pwl('no scale')
        pwl('feature_dim(C):{}'.format(self.d_args.var_dim))
        pwl('time_label_num:{}'.format(self.label_num))
        pwl('input_len: {}'.format(self.d_args.input_lag))
        pwl('output_len: {}'.format(self.d_args.horizon))
        pwl('border1(start id of train/val/test ds): {}'.format(self.d_args.border1))
        
        return obj_ds_dict, obj_stamp_dict
            
    
    def _data_scale(self, 
                    ts_df:pd.DataFrame,
                    train_end:int):

        s_o = ScaleOperator()
        
        scaler_key = self.e_args.scale_type

        scaled_data = s_o.ts_scale(ts_df,
                                   scaler_key,
                                   train_end) 
        scaler = s_o.scaler

        return scaled_data, scaler


    def _time_label(self,
                    df_stamp) -> np.ndarray:
        df_stamp['date'] = pd.to_datetime(df_stamp.datetime)
        freq_list = self.d_args.freq_list
        label_list = ['month','day','weekday','hour','15min','min','second']
        label_dict = {}
        for i,f in enumerate(freq_list):
            label_dict[f] = label_list[0:i+1]

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['15min'] = df_stamp.date.apply(lambda row: row.minute // 15, 1)
            df_stamp['min'] = df_stamp.date.apply(lambda row: row.minute, 1)
            
            label_cols =  label_dict[self.freq]
            data_stamp = df_stamp[label_cols].values
        
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # elif self.timeenc == 2:
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     data_stamp = df_stamp[['weekday']].values
        
        return data_stamp
        
        

class DsSplit(object):
    def __init__(self, d_args:DatasetArgs):
        self.border1 = [0,0,0]
        self.border2 = [0,0,0]
        self.sample_n_list = [0,0,0]

        self.d_args = d_args
        self.ts_len = d_args.ts_len
 
        self.input_lag = d_args.input_lag
        self.horizon = d_args.horizon
        self.sample_len = self.input_lag + self.horizon
        self.skip_len = d_args.skip_len


    def split_by_rate(self):  

        train_rate = self.d_args.train_rate
        val_rate = self.d_args.val_rate        

        total_sample_num = int((self.ts_len - self.sample_len)/self.skip_len) + 1
        num_train = int(total_sample_num * train_rate)
        num_val = int(total_sample_num * val_rate)
        num_test = int(total_sample_num - num_train - num_val)
        
        train_start = 0
        train_end = train_start + (self.sample_len-1) + self.skip_len*(num_train-1) 

        val_start = train_end - (self.sample_len-1) + self.skip_len
        val_end = val_start + (self.sample_len-1) + self.skip_len*(num_val-1)

        test_start = val_end - (self.sample_len-1) + self.skip_len
        test_end = self.ts_len - 1

        self.border1 = [train_start, val_start, test_start]
        self.border2 = [train_end+1, val_end+1, test_end+1] 
        self.sample_n_list = [num_train, num_val, num_test]


    def split_by_date(self, 
                      ts_start_date:str):
        start_date_list= self.d_args.start_date_list

        if start_date_list[0] == None:
            self.border1[0] = 0      # train start
            for i,flag in enumerate(self.d_args.flag_list[1:]):  # val,test start
                temp_range = pd.date_range(start=ts_start_date,
                                           end=start_date_list[i+1],
                                           freq=self.d_args.freq)
                self.border1[i+1] = len(temp_range) - 1
        else:
            for i,flag in enumerate(self.d_args.flag_list):
                temp_range = pd.date_range(start=ts_start_date,
                                           end=start_date_list[i],
                                           freq=self.d_args.freq)
                self.border1[i] = len(temp_range) - 1

        train_end = self.border1[1] - self.skip_len + (self.sample_len-1) 
        val_end = self.border1[2] - self.skip_len + (self.sample_len-1) 
        test_end = self.ts_len - 1
       
        num_train = int((self.border1[1] - self.border1[0])/self.d_args.skip_len)
        num_val = int((self.border1[2] - self.border1[1])/self.d_args.skip_len)
        num_test = int((test_end - val_end)/self.d_args.skip_len)

        self.border2 = [train_end+1, val_end+1, test_end+1]
        self.sample_n_list = [num_train, num_val, num_test]


    def generate_ds(self, 
                    data:np.ndarray, 
                    data_stamp:np.ndarray):
        '''
        data:(series_L, C); data_stamp:(series_L, label_num)
        '''
        all_data = np.concatenate((data, data_stamp), axis=1)
        '''
        all_data:(series_L, C + label_num)
        '''
        C = data.shape[1]

        ds_dict = {}        
        ds_stamp_dict = {}  
        for i,flag in enumerate(self.d_args.flag_list):
            ts_data = all_data[self.border1[i]:self.border2[i], :]
            sample_n = self.sample_n_list[i]

            ds_dict[flag] = self._cut_samples(ts_data,sample_n)[:,:,:C]
            ds_stamp_dict[flag] = self._cut_samples(ts_data,sample_n)[:,:,C:]
        
        '''
        L:sample_len;
        ds:(sample_num, L, C), ds_stamp:(sample_num, L, label_num)
        '''
        return ds_dict, ds_stamp_dict
        
    def _cut_samples(self,
                     ts_data:np.ndarray,
                     sample_n:int):
        '''
        ts_data:(series_L, C+label_num)
        '''
        # var_n = C+label_num
        var_n = ts_data.shape[1]
        ds_shape = (sample_n, self.sample_len, var_n)
        
        ds = np.zeros(ds_shape, dtype=np.float32)

        for i in range(0, sample_n):
            cut_start = i*self.skip_len
            cut_end = cut_start + self.sample_len
            sample = ts_data[cut_start:cut_end, :]
            
            ds[i,:] = sample
        
        '''
        ds:(sample_n, L= sample_len, C+label_num)
        '''
        return ds
        

    
    
