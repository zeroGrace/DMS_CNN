import numpy as np
import pandas as pd
import torch

import os
import sys
import gc
import datetime
root_path = os.path.dirname(os.path.dirname(__file__))  
sys.path.append(root_path)

from utils.tools import ExpErrors
from utils.tools import PrintWithLogging as PWL
from utils.tools import SaveDict
import utils.figs as figs

from exp.exp_args import ExpArgs
from exp.exp_main import Exp

from data_provider.data_args import DatasetArgs
from run_exp.d_args import D_Args
from data_provider.data_pre import DsPre

from models.model_args import Args_DMSCNN as M_args

gc.collect()

# ========== vars ==========
m_name = 'DMSCNN'

name_list = ['ECL321']
data_name = name_list[0]

d_args = D_Args(data_name).d_args_dict[data_name]

m_args = M_args(
    sight_list=[3,5,7],
    conv_ks_1 = 5,
    decomp_ks_list = [5, 15, 25, 45],
    trend_hidden_layers = [64, 64],
    dropout = 0.0,
    batch_size = d_args.batch_size, 
    epochs = 50, 
    lr = 0.001,
)

e_args = ExpArgs(
    repeat_time = 1,
    data_name = d_args.data_name,
    # path
    root_path = root_path, 
    # model setting
    forecast_mode = 'g',
    model_name = m_name,
    patience = 7,
    lradj = True,
    adj_type = 'type1',
    train_only = False,
    # series selection
    select_series = False,
    select_id_list = [],
    # scale
    is_scale = True,
    scale_type = 'minmax'
)

model_info = 'sight{}_cks{}_dks_{}_thl_{}_dr{}'.format(
    m_args.sight_list, m_args.conv_ks_1, 
    m_args.decomp_ks_list, m_args.trend_hidden_layers, m_args.dropout
)

seed = 2024

error_list = ['sMAPE', 'RMSE', 'MAE', 'MSE']

save_info = True

# ========== functions ==========
# ------ data_info ------
def gen_data_info(d_args:DatasetArgs, 
                  e_args:ExpArgs):
    data_info = '{}_{}_X{}_y{}_sk{}_ft{}_bs{}'.format(
    e_args.data_name, e_args.forecast_mode,
    d_args.input_lag, d_args.horizon, 
    d_args.skip_len, d_args.features_type, d_args.batch_size)
    
    if e_args.is_scale == True:
        data_info += '_{}'.format(e_args.scale_type)
    
    return data_info


# ------ exp_info ------
def get_exp_info(data_info, exp_time):
    if e_args.lradj == True:
        return data_info + '_{}_ep{}_lr{}_{}'.format(
            e_args.model_name, m_args.epochs, m_args.lr, exp_time)
    else:
        return data_info + '_{}_ep{}_lr{}_lrNoAdj_{}'.format(
            e_args.model_name, m_args.epochs, m_args.lr, exp_time)


# ------ seed ------
def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)        


def run_train(exp_info_path:str,
              exp:Exp,
              pwl:PWL):

    pwl('\n----------training : {}--------------------'.format(exp_info_path))
    train_loss, val_loss = exp.train(exp_info_path = exp_info_path)

    folder_path = os.path.join(
        e_args.fig_path, e_args.data_name, exp_info_path)
    figs.draw_loss(train_loss = train_loss,
                   valid_loss = val_loss,
                   save_path = folder_path)


def run_test(exp_info_path:str,
             exp:Exp,
             pwl:PWL,
             obj_id:int = None):
    
    pwl('\n----------testing : {}--------------------'.format(exp_info_path))
    result_save_path = exp.test(exp_info_path = exp_info_path,
                                obj_id= obj_id)

    torch.cuda.empty_cache()

    return result_save_path
    

# ========== run exp ==========
def main():
    time_key = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    pwl = PWL()
    
    data_info = gen_data_info(d_args, e_args)

    exp_info = get_exp_info(data_info = data_info,
                            exp_time = time_key)

    if save_info == True:
        sd = SaveDict(time_key = time_key)
        sd.save_exp_dict(data_info, exp_info)

    pwl.log_config(e_args, exp_info, time_str = time_key)

    seed_all(seed)

    pwl('device: {}'.format(e_args.device))
    pwl('model details:{}_{}_{}'.format(
        e_args.forecast_mode, e_args.model_name, model_info))
    pwl('random seed:{}'.format(seed))
    
    ds_pre = DsPre(d_args = d_args,
                   e_args = e_args)
    pwl('\n>>>>>>>get dataset : {}<<<<<<<<<<<<<<<<<<<<<'.format(data_info))
    
    multi_ds_dict = ds_pre.multi_ds_dict
    
    if save_info == True:
        sd.save_args_instance(d_args = d_args, 
                              m_args = m_args)

    # -------- local --------
    if e_args.forecast_mode == 'l':

        # --- repeat exp ---
        for t in range(e_args.repeat_time):
            pwl('\n~~~~~~~~~~~~~~  start of rt{} ~~~~~~~~~~~~~~'.format(t))
            
            ee = ExpErrors(obj_num = d_args.obj_num,
                           error_list = error_list,
                           e_args = e_args,
                           exp_info = exp_info,
                           rt = t)

            for obj_id in e_args.ts_id_list:

                pwl('\n~~~~~~~  start of obj_id{} ~~~~~~~'.format(obj_id))

                # key:train/valid/test； value: 2d ndarray
                ds_dict = multi_ds_dict[obj_id]

                exp_info_path = os.path.join(
                    exp_info, 'rt{}'.format(t), 'obj{}'.format(obj_id))
        
                exp = Exp(d_args = d_args,
                          m_args = m_args,
                          e_args = e_args,
                          ds_dict = ds_dict)
                
                run_train(exp_info_path, exp, pwl)
                
                result_save_path = run_test(exp_info_path, exp, pwl, obj_id)

                ee.local_series_error(obj_id = obj_id,
                                      result_save_path = result_save_path)

                pwl('\n~~~~~~~ end of obj_id{} ~~~~~~~'.format(obj_id))
            
            ee.save_series_error(series_index = e_args.ts_id_list)
            ee.total_error()

            pwl('\n~~~~~~~~~~~~~~  end of rt{} ~~~~~~~~~~~~~~'.format(t))


    # -------- global --------
    else:
        assert e_args.forecast_mode == 'g'
        
        ds_dict = multi_ds_dict['g']
        
        # --- repeat exp ---
        for t in range(e_args.repeat_time):
            pwl('\n~~~~~~~~~~~~~~  start of rt{} ~~~~~~~~~~~~~~'.format(t))
            
            ee = ExpErrors(obj_num = d_args.obj_num,
                           error_list = error_list,
                           e_args = e_args,
                           exp_info = exp_info,
                           rt = t)

            exp_info_path = os.path.join(exp_info, 'rt{}'.format(t))


            exp = Exp(d_args=d_args,
                      m_args=m_args,
                      e_args=e_args,
                      ds_dict=ds_dict)
            
            run_train(exp_info_path, exp, pwl)
            
            result_save_path = run_test(exp_info_path, exp, pwl)

            ee.global_series_error(result_save_path)
            ee.save_series_error(series_index = e_args.ts_id_list)
            ee.total_error()            
            
            pwl('\n~~~~~~~~~~~~~~  end of rt{} ~~~~~~~~~~~~~~'.format(t))



if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

