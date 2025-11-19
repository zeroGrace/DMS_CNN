'''
Copyright (C) 2025 Siyue Yang, Yukun Bao
<siyue_yang@hust.edu.cn>, <yukunbao@hust.edu.cn>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
'''

import numpy as np
import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
from models_dl.blocks import MLP
from layers.decomp import series_decomp_multi

class PatchConvBlock(nn.Module):
    def __init__(self, 
                 horizon,
                 lag_day_num, 
                 pred_day_num,
                 day_point_n, 
                 sight, 
                 ks_1):
        
        super().__init__()
        
        self.lag_day_num = lag_day_num
        self.horizon = horizon
        self.pred_day_num = pred_day_num
        self.day_point_n = day_point_n
        
        self.sight = sight
        self.conv_num = lag_day_num - sight + 1
        
        p_0 = 0
        ks_0 = sight
        ks_1 = ks_1
        p_1 = int((ks_1-1)/2)
        self.convList = torch.nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels = 1, 
                    out_channels = self.pred_day_num,
                    kernel_size = (ks_0, ks_1),
                    stride = 1,
                    padding = (p_0, p_1),
                    padding_mode = 'circular'
                ) for i in range(self.conv_num)
            ]
        )
        
        in_dim = self.conv_num * horizon
        out_dim = horizon
        hidden_layers = [64,]
        self.projection = MLP(in_dim, hidden_layers, out_dim)
        
        
    def forward(self, x):
        B = x.shape[0]
        C = x.shape[1]
        x = x.view(B, C, self.lag_day_num, self.day_point_n)
        
        conv_output = torch.cat([self.convList[i](x[:,:,i:i+self.sight,:]).view(B, C, -1, self.pred_day_num*self.day_point_n)
                                 for i in range(self.conv_num)], dim=2)
        
        x = self.projection(conv_output.view(B, C, -1))
        
        return x
        
        
        
class PeriodModule(nn.Module):
    def __init__(self, 
                 lag_day_num, 
                 day_point_n, 
                 horizon, 
                 sight_list,
                 ks_1):
        
        super().__init__()

        pred_day_num = int(horizon/day_point_n)

        self.sight_list = sight_list
        
        self.patchBlock_list = torch.nn.ModuleList(
            PatchConvBlock(
                horizon,
                lag_day_num, 
                pred_day_num,
                day_point_n, 
                sight, 
                ks_1
            ) for sight in sight_list
        )

        in_dim = int(len(sight_list))*horizon
        out_dim = horizon
        hidden_layers = [64,]
        self.projection = MLP(in_dim, hidden_layers, out_dim)


    def forward(self, x):
        x = x.permute(0,2,1)   
        B = x.shape[0]
        C = x.shape[1]
        
        mulit_sight_results = torch.cat([self.patchBlock_list[i](x) for i in range(len(self.sight_list))], dim = 1)
        
        x = self.projection(mulit_sight_results.view(B, C, -1)).permute(0,2,1)
        
        return x
    


class Model(nn.Module):
    def __init__(self, m_args, d_args):
        
        super().__init__()

        day_point_n = d_args.day_point_n
        lag_day_num = int(d_args.input_lag / day_point_n)
        horizon = d_args.horizon
        
        decomp_ks_list = m_args.decomp_ks_list
        self.decomp = series_decomp_multi(decomp_ks_list)
        
        self.trend_module = MLP(in_dim = d_args.input_lag, 
                                out_dim = horizon, 
                                hidden_layers = m_args.trend_hidden_layers)
        
        self.preiod_module = PeriodModule(
            lag_day_num, 
            day_point_n, 
            horizon, 
            m_args.sight_list,
            m_args.conv_ks_1
        )
        
        
    def forward(self, x):
        sea, moving_mean = self.decomp(x)
        
        trend = self.trend_module(moving_mean.permute(0,2,1)).permute(0,2,1)
        x = self.preiod_module(sea) + trend
        
        return x