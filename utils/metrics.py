import numpy as np

'''
true,pred shape:(n_sample, L, C)
'''

def sAPE(pred, true):
    return 2 * np.abs(pred - true) / (np.abs(true) + np.abs(pred))
    
def AE(pred, true):
    return np.abs(pred - true)
    
def SE(pred, true):
    return (pred-true)**2 

def sMAPE(pred, true):
    return np.mean(2 * np.abs(pred - true) / (np.abs(true) + np.abs(pred)))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def get_error(pred, true, error_list):
    # key: error name; value: error func name
    error_func_dict = {
        'sAPE': sAPE,
        'AE': AE,
        'SE': SE,
        'sMAPE': sMAPE,
        'MAPE': MAPE,
        'MAE':MAE,
        'MSE':MSE,
        'RMSE':RMSE,
    }
    
    return np.array([error_func_dict[k](pred, true) for k in error_list])
