# Global electricity demand forecasting for multi-consumer retailers using a decomposition-based multi-sight convolutional neural network

Siyue Yang, Qi Sima, Liang Shen, Yukun Bao*

*Center for Modern Information Management, School of Management, Huazhong University of Science and Technology, Wuhan, 430074 CN

This is the PyTorch implementation of the proposed method in the paper "Global electricity demand forecasting for multi-consumer retailers using a decomposition-based multi-sight convolutional neural network", which has been accepted by Computers in Industry.

## Requirements
- Python 3.8
- matplotlib == 3.9.2
- numpy == 1.26.4
- pandas == 2.2.2
- torch == 2.6.0

## Project
- **Core Model Files**  
  - `models/models_dl/dmsCNN.py` - Model architecture of the proposed DMS-CNN. 
  - `models/model_args.py` - Key hyperparameters of DMS-CNN. 
- **Program Entry File** 
  - `run_exp/run_exp_dmsCNN.py` - Launches the experiment and generates the results.  


## Acknowledgement
This work was supported by the National Natural Science Foundation of China (72242104) and the Interdisciplinary Research Program of Huazhong University of Science and Technology (2024JCYJ020).

## Citation

```
@article{yangetalGlobal2026,
  title = {Global electricity demand forecasting for multi-consumer retailers using a decomposition-based multi-sight convolutional neural network},
  author = {Yang, Siyue and Sima, Qi and Shen, Liang and Bao, Yukun},
  year = 2026,
  journal = {Computers in Industry},
  volume = {174},
  pages = {104415}
}
```

## Contact
If you have any questions, welcome to contact us: 
- Siyue Yang (Ph.D. student, siyue_yang@hust.edu.cn)
- Yukun Bao (Professor, yukunbao@hust.edu.cn)



