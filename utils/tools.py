import pandas as pd
import numpy as np
import torch

import sys
import os
import logging
# import datetime
import json
# import re

sys.path.append(os.path.dirname(__file__))

import metrics as m

class AdjustLR(object):
    def __init__(self, 
                 lr_init:float,
                 lradj:bool = True,
                 adj_type:str = 'type1'):
        
        self.lr_init = lr_init
        self.lradj = lradj
        self.adj_type = adj_type
        

    def __call__(self, optimizer, epoch):
        print_with_log = PrintWithLogging()

        if self.lradj:
            if self.adj_type == 'type1':
                lr_adjust = {epoch: self.lr_init * (0.5 ** (epoch // 1))}
            
            elif self.adj_type == 'type2':
                lr_adjust = {
                    1: 5e-5, 3: 1e-5, 5: 5e-6, 7: 1e-6, 
                    9: 5e-7, 14: 1e-7, 19: 5e-8
                }
        
            if epoch in lr_adjust.keys():
                lr = lr_adjust[epoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print_with_log('Updating learning rate to {}'.format(lr))
        
        else:
            pass



class EarlyStopping(object):
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.valid_loss_min = np.Inf
        self.delta = delta

    def __call__(self, valid_epoch_loss, model, save_path):
        print_with_log = PrintWithLogging()
        
        score = -valid_epoch_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_epoch_loss, model, save_path)
        
        elif score < self.best_score + self.delta:
            
            self.counter += 1
            print_with_log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(valid_epoch_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, valid_loss, model, save_path) -> None:
        print_with_log = PrintWithLogging()
        
        if self.verbose:
            print_with_log(f'Validation loss decreased ({self.valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), os.path.join(save_path , 'checkpoint.pth'))
        self.valid_loss_min = valid_loss



class SaveDict(object):
    def __init__(self,
                 time_key:str) -> None:
        self.path = os.path.dirname(__file__)
        self.path_nc = os.path.join(os.path.dirname(self.path), '_NC')
        
        self.time_key = time_key

    
    def _set_dict(self,
                  json_file_name:str):
        
        save_path = self.path_nc
        if not os.path.exists(save_path):
            os.makedirs(save_path)    
        dict_file_path = os.path.join(save_path, json_file_name)
        
        if os.path.isfile(dict_file_path):
            with open(dict_file_path, 'r') as file:
                dict_save = json.load(file)
        else:
            dict_save = {}
        
        return dict_save, dict_file_path
    
    
    def _save_dict(self,
                   dict_save:dict,
                   dict_file_path:str):
        
        with open(dict_file_path, 'w') as file:
            json.dump(dict_save, file)
    
            
    def save_exp_dict(self,
                      data_info:str, 
                      exp_info:str):

        exp_dict, exp_dict_path = self._set_dict(json_file_name = 'exp_dict.json')
            
        exp_dict[self.time_key] = [data_info, exp_info]

        self._save_dict(dict_save = exp_dict,
                        dict_file_path = exp_dict_path)
    
        
    def save_args_instance(self, 
                           **kwargs_insts):
        args_dict_allexp, args_dict_path = self._set_dict(json_file_name = 'args.json')
        args_dict = {}
        
        for k,v in kwargs_insts.items():
            args_dict[k] = v.__dict__  
        
        args_dict_allexp[self.time_key] = args_dict
        
        self._save_dict(dict_save = args_dict_allexp,
                        dict_file_path = args_dict_path)     
        
                                                    

class ExpErrors():
    def __init__(self,
                 obj_num:int,
                 error_list:list,
                 e_args,
                 exp_info:str,
                 rt:int) -> None:
        self.error_list = error_list
        self.obj_num = obj_num
        
        self.se = np.zeros((obj_num, len(error_list))) 
        
        _half_path = os.path.join(e_args.result_path, e_args.data_name)
        self.save_path = os.path.join(_half_path, exp_info, 'rt{}'.format(rt))
        
        self.is_scale = e_args.is_scale
        if self.is_scale:
            self.scaled_se = np.zeros((obj_num, len(error_list)))

        self.scale_type = e_args.scale_type
            

    def local_series_error(self,
                           obj_id:int,
                           result_save_path:str):
        preds = np.load(os.path.join(result_save_path, 'preds.npy'))
        trues = np.load(os.path.join(result_save_path, 'trues.npy'))
        self.se[obj_id] = m.get_error(preds, trues, self.error_list)

        if self.is_scale:
            scaled_preds = np.load(os.path.join(result_save_path, 
                                                'scaled_preds.npy'))
            scaled_trues = np.load(os.path.join(result_save_path, 
                                                'scaled_trues.npy'))

            self.scaled_se[obj_id] = m.get_error(scaled_preds, 
                                              scaled_trues, 
                                              self.error_list)
    
    
    def global_series_error(self,
                            result_save_path:str):      
        preds = np.load(os.path.join(result_save_path, 'preds.npy'))
        trues = np.load(os.path.join(result_save_path, 'trues.npy'))
        
        if self.is_scale:
            scaled_preds = np.load(os.path.join(result_save_path, 
                                                'scaled_preds.npy'))
            scaled_trues = np.load(os.path.join(result_save_path, 
                                                'scaled_trues.npy'))
        
        for id in range(self.obj_num):
            self.se[id] = m.get_error(preds[id], trues[id], self.error_list)
            
            if self.is_scale:
                self.scaled_se[id] = m.get_error(scaled_preds[id], 
                                                  scaled_trues[id], 
                                                  self.error_list)
        

    def save_series_error(self, series_index):
        se_file = os.path.join(self.save_path, 'series_error.csv')
        pd.DataFrame(self.se, 
                     columns = self.error_list, 
                     index = series_index).to_csv(se_file)
        
        if self.is_scale:
            scaled_se_file = os.path.join(self.save_path, 'scaled_series_error.csv')
            pd.DataFrame(self.scaled_se, 
                         columns = self.error_list, 
                         index = series_index).to_csv(scaled_se_file)


    def total_error(self):
        
        total_error = np.mean(self.se, axis = 0, keepdims = True)

        te_file = os.path.join(self.save_path, 'total_error.csv')
        pd.DataFrame(total_error, 
                     columns = self.error_list, 
                     index = ['value']).to_csv(te_file)
        
        if self.is_scale:
            scaled_te = np.mean(self.scaled_se, axis  =0, keepdims = True)
            scaled_te_file = os.path.join(self.save_path, 'scaled_total_error.csv')
            pd.DataFrame(scaled_te, 
                         columns = self.error_list, 
                         index = ['value']).to_csv(scaled_te_file)
    
        
        
class PrintWithLogging(object):
    def __init__(self) -> None:
        pass
    
    def log_config(self,
                   e_args, 
                   exp_info:str,
                   time_str:str):
    
        log_folder = os.path.join(e_args.result_path, e_args.data_name, exp_info)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            
        log_name = f"log_{time_str}.log"

        log_file = os.path.join(log_folder, log_name)

        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
    
    def __call__(self, info) -> None:
        print(info)
        logging.info(info)  
        
    

class DataReshape(object):
    def __init__(self) -> None:
        pass
    
    def add_serie_dim(data_g:np.ndarray,
                      obj_num:int) -> np.ndarray:
        assert data_g.shape[0] % obj_num == 0
        
        data = data_g.reshape((obj_num, 
                               int(data_g.shape[0]//obj_num), 
                               data_g.shape[-2],
                               data_g.shape[-1]))
        return data



class DictToClass:
    def __init__(self, input_dict):
        self._dict = input_dict
        for key, value in input_dict.items():
            setattr(self, key, value)