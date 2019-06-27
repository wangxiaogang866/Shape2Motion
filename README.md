# Shape2Motion: Joint Analysis of Motion Parts and Attributes from 3D Shapes

This is the code repository for "Shape2Motion: Joint Analysis of Motion Parts and Attributes from 3D Shapes”

Created by Xiaogang Wang, Bin Zhou, Yahao Shi, Xiaowu Chen, Qinping Zhao, Kai Xu


## Prerequisites: 
    Numpy (ver. 1.13.3)
    TensorFlow (ver. 1.4.1)
    scipy (ver. 0.19.1)
     
## Dataset 
[June 13, 2019] We release our 3D object mobility dataset 'Motion Dataset V0' [here](http://motiondataset.zbuaa.com/).

You can download the data (point cloud data) [here](http://www.zbuaa.com/CVPR19/dataset.zip) for training and testing.


## Train：

    to train Motion Part Proposal Module and Motion Attribute Proposal Module

    python train_stage_12 --stage=1

    to train a Proposal Matching Module

    python train_stage_12 --stage=2 --batch_size=32

    to train a Motion Optimizatoin Network

    python train_stage_3

## Test：

    to test Motion Part Proposal Module and Motion Attribute Proposal Module

    python test_stage_12 --stage=1

    to test a Proposal Matching Module

    python test_stage_12 --stage=2 --batch_size=32

    to test a Motion Optimizatoin Network

    python test_stage_3

## Other:
    nms.m: This file is used for merging proposal.
    generate_stage_2_train_data.m: This file is used for generating stage 2 training_data.
    generate_stage_3_train_data.m: This file is used for generating stage 3 training_data.
    
## Evaluation:
    evaluation.m: This file is used for part mobility metrics.

## Citation

If you find our paper useful in your research, please cite:

 @article{wang_siga18,\
   title = {Shape2Motion: Joint Analysis of Motion Parts and Attributes from 3D Shapes},\
   author = {Xiaogang Wang and Bin Zhou and Yahao Shi and Xiaowu Chen and Qinping Zhao and Kai Xu},\
   journal = {IEEE Conference on Computer Vision and Pattern},\
   volume = {XX},\
   number = {XX},\
   pages = {to appear},\
   year = {2019}\
  }
