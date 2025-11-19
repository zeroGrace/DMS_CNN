import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_provider.data_args import DatasetArgs

class D_Args(object):
    def __init__(self, data_name, **kwargs):
        day_points_dict = {
            'h': 24,
            '15t': 96,
            't': 24*60,
        }
        
        self.d_args_dict = {
            'ECL321': DatasetArgs(
                        data_name = data_name,
                        input_lag = 7 * day_points_dict['h'],
                        horizon = 1 * day_points_dict['h'],
                        label_len = kwargs['label_len'] if kwargs else None,
                        split_type = 'rate',
                        # split by rate
                        train_rate = 0.6,
                        val_rate = 0.2,
                        skip_len = int( day_points_dict['h']),
                        features_type = 'S',
                        freq = 'h',
                        batch_size = 32
                        ),  
        }
        