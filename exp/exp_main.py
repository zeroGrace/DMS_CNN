import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.tools import EarlyStopping, AdjustLR, PrintWithLogging, DataReshape
import utils.metrics as metrics

from data_provider.data_args import DatasetArgs
from data_provider.dataset_torch import DatasetDL as DatasetTorch

from models.model_args import ModelArgs
from models.models_dl import dmsCNN
from exp.exp_args import ExpArgs

class Exp():
    def __init__(self, 
                 d_args:DatasetArgs, 
                 m_args:ModelArgs,
                 e_args:ExpArgs,
                 ds_dict:dict):
        
        '''
        ds_dict: 
            - key: 'train'/'val'/'test'; 
            - value: (sample_num, L, C)
        '''
        
        self.d_args = d_args
        self.m_args = m_args
        self.e_args = e_args
        self.ds_dict = ds_dict

        self.exp_metrics_scaled = None
        self.exp_metrics = None

        self.train_time = None
        self.epoch_num = None
        
        self.f_dim = -1 if self.d_args.features_type == 'MS' else 0
        
        self.model = self._build_model().to(self.e_args.device)
    
      
    def _get_data(self, flag):
        
        assert flag in self.d_args.flag_list
        
        # data_set: torch.Dataset
        data_set = DatasetTorch(self.ds_dict[flag], self.d_args)
        
        # data_loader: torch.DataLoader
        if flag == 'test':
            data_loader = DataLoader(dataset=data_set,
                                     batch_size=self.d_args.batch_size,
                                     drop_last=False,
                                     shuffle=False)
        else:
            data_loader = DataLoader(dataset=data_set,
                                     batch_size=self.d_args.batch_size,
                                     drop_last=False,
                                     shuffle=True)
        
        return data_set, data_loader
    
    
    def _build_model(self): 
        
        model_dict = {
            'DMSCNN':dmsCNN,
        }
            
        model = model_dict[self.e_args.model_name].Model(self.m_args, self.d_args).float()
        
        return model
    

    def _set_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr = self.m_args.lr)
        return optimizer
    
    
    def _set_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    
    
    def vali(self, val_loader, criterion):
        batch_loss = []
        
        self.model.eval()
        with torch.no_grad():
            for idx,(batch_x,batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(torch.float32).to(self.e_args.device)
                
                outputs = self.model(batch_x)
                
                outputs = outputs[:, :, self.f_dim:]
                batch_y = batch_y[:, :, self.f_dim:].to(torch.float32).to(self.e_args.device)
                
                loss = criterion(batch_y,outputs)
                
                batch_loss.append(loss.item())
    
        epoch_loss = np.average(batch_loss)
        
        self.model.train()
        
        return epoch_loss
            

    def train(self, exp_info_path):
        pwl = PrintWithLogging()
        
        train_ds, train_loader = self._get_data(flag = 'train')
        if not self.e_args.train_only:
            val_ds, val_loader = self._get_data(flag = 'val')
            test_data, test_loader = self._get_data(flag='test')
        
        save_path = os.path.join(self.e_args.checkpoints, self.e_args.data_name, exp_info_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)                                      
        
        time_now = time.time()
        time_start = time.time()
        
        train_steps = len(train_loader)
        
        early_stopping = EarlyStopping(patience=self.e_args.patience, 
                                       verbose=True)
        
        adjust_lr = AdjustLR(lr_init =self.m_args.lr,
                             lradj = self.e_args.lradj,
                             adj_type=self.e_args.adj_type)
        
        optimizer = self._set_optimizer()
        criterion =  self._set_criterion()
        train_loss = []
        val_loss = []
        
        epoch_num = 0 
        
        for epoch in range(self.m_args.epochs):
            iter_count = 0
            
            train_batch_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for idx,(batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                batch_x = batch_x.to(torch.float32).to(self.e_args.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                
                outputs = outputs[:, :, self.f_dim:]
                batch_y = batch_y[:, :, self.f_dim:].to(torch.float32).to(self.e_args.device)
            
                loss = criterion(batch_y,outputs)
                train_batch_loss.append(loss.item())
                
                loss.backward()

                optimizer.step()
                
                if self.e_args.forecast_mode == 'l':
                    pass
                else:
                    if (idx+1) % (train_steps//2) == 0:
                        pwl("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(idx, epoch, loss.item()))
                        
                        speed = (time.time()-time_now)/iter_count
                        
                        left_time = speed*((self.m_args.epochs - epoch)*train_steps - idx)
                        
                        pwl('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        
                        iter_count = 0
                        time_now = time.time()
            
            pwl("Epoch: {} cost time: {:.4f}s".format(epoch, time.time()-epoch_time))    
            
            train_loss.append(np.average(train_batch_loss))
            
            with torch.no_grad():
                val_loss.append(self.vali(val_loader, criterion))
            
            pwl("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch, train_steps, train_loss[-1], val_loss[-1]))
            
            early_stopping(val_loss[-1], self.model, save_path)
            if early_stopping.early_stop:
                pwl("Early stopping")
                break

            adjust_lr(optimizer, epoch)    
            
            epoch_num += 1
            
        time_end = time.time()
        
        self.train_time = time_end-time_start
        self.epoch_num = epoch_num
        
        pwl("====== total training time: {:.4f}s, epoch_num: {} ======".format(
            time_end-time_start, epoch_num
            ))
        
        best_model= os.path.join(save_path, 'checkpoint.pth')
        
        self.model.load_state_dict(torch.load(best_model))
        
        return train_loss, val_loss


    def test(self, exp_info_path, obj_id = None):
        
        pwl = PrintWithLogging()
        
        test_ds, test_loader = self._get_data(flag = 'test')
        
        save_path = os.path.join(self.e_args.checkpoints, 
                                 self.e_args.data_name, 
                                 exp_info_path)
        best_model= os.path.join(save_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model))
        
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for idx,(batch_x,batch_y) in enumerate(test_loader):
                batch_x = batch_x.to(torch.float32).to(self.e_args.device)
                batch_y = batch_y.to(torch.float32).to(self.e_args.device)
                
                outputs = self.model(batch_x)  
                
                outputs = outputs[:, :, self.f_dim:]
                batch_y = batch_y[:, :, self.f_dim:]  

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        if self.e_args.forecast_mode == 'g':
            preds = DataReshape.add_serie_dim(preds, self.d_args.obj_num)
            trues = DataReshape.add_serie_dim(trues, self.d_args.obj_num)
        
        pwl('shape of preds:{}; trues:{}'.format(preds.shape, trues.shape))

        folder_path = os.path.join(self.e_args.result_path, 
                                   self.e_args.data_name, 
                                   exp_info_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        rmse = metrics.RMSE(pred = preds, true = trues)
        smape = metrics.sMAPE(pred = preds, true = trues)
        mae = metrics.MAE(pred = preds, true = trues)
        
        self.exp_metrics_scaled = [rmse, mae, smape]
        pwl('scaled RMSE:{:.6f}, MAE:{:.6f}, sMAPE:{:.6f}'.format(rmse, mae, smape))

        if self.e_args.is_scale == False:
            np.save(os.path.join(folder_path, 'preds.npy'), preds)
            np.save(os.path.join(folder_path, 'trues.npy'), trues)
        
        else:
            np.save(os.path.join(folder_path, 'scaled_preds.npy'), preds)
            np.save(os.path.join(folder_path, 'scaled_trues.npy'), trues)

            # descale
            preds_origin, trues_origin = self._descale_preds_trues(trues, preds, obj_id)

            rmse = metrics.RMSE(pred = preds_origin,true = trues_origin)
            smape = metrics.sMAPE(pred = preds_origin,true = trues_origin)
            mae = metrics.MAE(pred = preds_origin,true = trues_origin)
            
            self.exp_metrics = [rmse, mae, smape]
            pwl('based on original data - RMSE:{:.6f}, MAE:{:.6f}, sMAPE:{:.6f}'.format(rmse, mae, smape))
            
            np.save(os.path.join(folder_path, 'preds.npy'), preds_origin)
            np.save(os.path.join(folder_path, 'trues.npy'), trues_origin)

        return folder_path
    

    # descale
    def _descale_preds_trues(self, trues, preds, obj_id = None):
        # -- init --
        preds_origin = np.zeros(preds.shape)
        trues_origin = np.zeros(trues.shape)
        
        # ---global---
        if self.e_args.forecast_mode == 'g':
            scaler_dict = self.e_args.scaler_dict
            '''
            trues,preds:(n_obj, n_sample, L, target_C)
            '''
            for i in range(trues.shape[0]):
                obj_id = self.e_args.ts_id_list[i]

                for sample_id in range(trues.shape[1]):
                    preds_origin[i][sample_id] = scaler_dict[obj_id].inverse_transform(preds[i][sample_id],
                                                                                       self.d_args.features_type)
                    trues_origin[i][sample_id] = scaler_dict[obj_id].inverse_transform(trues[i][sample_id],
                                                                                       self.d_args.features_type)

        # ---local---
        else:
            scaler = self.e_args.scaler_dict[obj_id]
            '''
            trues,preds:(n_sample, L, target_C)
            '''
            for sample_id in range(trues.shape[0]):
                preds_origin[sample_id] = scaler.inverse_transform(preds[sample_id], self.d_args.features_type)
                trues_origin[sample_id] = scaler.inverse_transform(trues[sample_id], self.d_args.features_type)
        
        return preds_origin, trues_origin
        
    

        
            
        
    
    
