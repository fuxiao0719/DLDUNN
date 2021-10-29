# DLDUNN
This repository contains the code for [Hybrid Precoding Design Based on Dual-Layer Deep-Unfolding Neural Network](https://ieeexplore.ieee.org/document/9569633) (IEEE PIMRC 2021)

## Requirements
* Python >= 3.6
* PyTorch >= 1.1.0

## Usage
* Run the  penalty dual decomposition (PDD) algorithm on complex Gaussian MIMO channel
```bash
python PDD.py
```
* Run the DLDUNN model
```bash
python run.py
```
* For easier verification, we set K=2, M=2, N=4, M_RF=2, N_RF=4 here. And we can achieve PDD's result(16.125bps/Hz) and DLDUNN's result(15.901bps/Hz, ~98.6\%). These parameters can be manually changed to match the paper.

## Cite
```
@INPROCEEDINGS{9569633,
  author={Zhang, Guangyi and Fu, Xiao and Hu, Qiyu and Cai, Yunlong and Yu, Guanding},
  booktitle={2021 IEEE 32nd Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)}, 
  title={Hybrid Precoding Design Based on Dual-Layer Deep-Unfolding Neural Network}, 
  year={2021},
  pages={678-683},
  doi={10.1109/PIMRC50174.2021.9569633}}
```