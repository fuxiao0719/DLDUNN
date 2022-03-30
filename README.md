# Hybrid Precoding Design Based on Dual-Layer Deep-Unfolding Neural Network
This repository contains the code for [Hybrid Precoding Design Based on Dual-Layer Deep-Unfolding Neural Network](https://ieeexplore.ieee.org/document/9569633) (IEEE PIMRC 2021). Slides can be seen [here](https://fuxiao0719.github.io/assets/pdf/pimrc_dldunn.pdf).

## Pipeline
<img src="pimrc2021.jpg" width = "100%"/>

## Requirements
* Python >= 3.6
* PyTorch >= 1.1.0

## Usage
* Run the  penalty dual decomposition (PDD) algorithm on complex Gaussian MIMO channel
```bash
python pdd.py --cfg_file lib/dldunn.yaml K 2 M 2 N 4 M_RF 2 N_RF 4
```
* Run the DLDUNN model
```bash
python run.py --cfg_file lib/dldunn.yaml K 2 M 2 N 4 M_RF 2 N_RF 4 D_in 4 D_out 2
```
* For easier verification, if we set K=2, M=2, N=4, M_RF=2, N_RF=4, D_in=4, D_out=2 here, we can achieve PDD's result(16.177bps/Hz) and DLDUNN's result(15.891bps/Hz). These parameters can be manually changed to match the paper.

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
