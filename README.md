# SCNet Code

This code is written in MATLAB, and implements the SCNet[1]. For the dataset, see our project page: http://www.di.ens.fr/willow/research/scnet.


# Install Dependencies
  - Install [MatConvNet] (http://www.vlfeat.org/matconvnet/).
  - Download [VLFeat] (http://www.vlfeat.org/) [0.9.20] in './utils/'
  - Download the following source codes of object proposal or other proposal methods you would like to test:
    - [Randomized Prim’s] (https://github.com/smanenfr/rp#rp) (our preference);
    - [EdgeBox] (https://github.com/pdollar/edges);
    - [SelectiveSearch] (http://koen.me/research/selectivesearch/);
    - [Multiscale Combinatorial Grouping] (https://github.com/jponttuset/mcg);
  - Download a ImageNet [Caffe Reference model] (http://www.vlfeat.org/matconvnet/pretrained/) in `./data/models/`. 

# Codes

## SCNet_Matconvnet

Additional Matconvnet modules implemented for SCNet. These code should be copied into `matconvnet/matlab/` folder.

## SCNet

This is the primary net work training and testing code. 

- `SCNet_A_init.m`, `SCNet_AG_init.m`, `SCNet_AGplus_init.m`: initialize the SCNet_A, SCNet_AG, SCNet_AG+.

- `SCNet_A.m`, `SCNet_AG.m`, `SCNet_AGplus.m`: train SCNet_A, SCNet_AG, SCNet_AG+.

- `eva_PCR_mIoU_SCNet_A.m`,  `eva_PCR_mIoU_SCNet_AG.m`, `eva_PCR_mIoU_SCNet_AGplus.m`: evaluate the trained nets.

- `eva_PCR_mIoU_ImageNet_SCNet_A.m`,  `eva_PCR_mIoU_ImageNet_SCNet_AG.m`, `eva_PCR_mIoU_ImageNet_SCNet_AGplus.m`: evaluate SCNets with ImageNet pretrained parameters, i.e., SCNets without training.

## SCNet_Baselines

Comparison code for our SCNet features and HOG features with NAM, PHM and LOM in Proposal Flow [2, 3].

- `NAM_HOG_eva.m`, `PHM_HOG_eva.m`, `LOM_HOG_eva.m`: evaluate NAM, PHM, and LOM with HOG features.
	
- `NAM_SCNet_eva.m`, `PHM_SCNet_eva.m`, `LOM_SCNet_eva.m`: evaluate NAM, PHM, and LOM with learned SCNet features.
 
- `HOG_SCNet_AG_eva.m`: replace the learned SCNet feature by HOG feature in SCNet_AG model.

## Data
We used PF-PASCAL, PF-WILLOW, PASCAL Parts and CUB data sets and follows Proposal Flow[2, 3] to generate our trainging data.

  
# Notes

  - The code is provided for academic use only. Use of the code in any commercial or industrial related activities is prohibited. 
  - If you use our code or dataset, please cite the paper. 

```
@InProceedings{khan2017,
author = {Kai Han, Rafael S. Rezende, Bumsub Ham, Kwan-Yee K. Wong, Minsu Cho, Cordelia Schmid, and Jean Ponce},
title = {SCNet: Learning Semantic Correspondence},
booktitle = {arXiv:1705.04043},
year = {2017}
}
```

  
# References

[1] Kai Han, Rafael S. Rezende, Bumsub Ham, Kwan-Yee K. Wong, Minsu Cho, Cordelia Schmid, Jean Ponce,  "SCNet: Learning Semantic Correspondence", *International Conference on Computer Vision (ICCV)*, 2017.

[2] Bumsub Ham, Minsu Cho, Cordelia Schmid, Jean Ponce, "Proposal Flow: Semantic Correspondences from Object Proposals", *IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI)*, 2017 

[3] Bumsub Ham, Minsu Cho, Cordelia Schmid, Jean Ponce, "Proposal Flow", *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016 
