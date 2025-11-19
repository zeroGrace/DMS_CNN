import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def pic_save(fig,
             pic_name:str, 
             save_path:str):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig.savefig(os.path.join(save_path, pic_name + '.png'))
    fig.savefig(os.path.join(save_path, pic_name + '.svg'))
    
    
def draw_loss(train_loss:list, 
              valid_loss:list, 
              save_path:str):
    
    
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].plot(train_loss)
    axs[0].set_title("epochs_loss_train")
    
    axs[1].plot(train_loss,'-o',label="train_loss")
    axs[1].plot(valid_loss,'-o',label="valid_loss")
    axs[1].set_title("epochs_loss_train_valid")

    fig.legend()
    
    pic_name = 'training_loss'
    pic_save(fig=fig,
             pic_name=pic_name, 
             save_path=save_path)
    
    




    