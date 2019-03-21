# Shape2Motion

Requirements：
    Numpy (ver. 1.13.3)
    TensorFlow (ver. 1.4.1)
    scipy (ver. 0.19.1)

Usage：
Train：

    to train Motion Part Proposal Module and Motion Attribute Proposal Module

    python train_stage_12 --stage=1

    to train a Proposal Matching Module

    python train_stage_12 --stage=2 --batch_size=32

    to train a Motion Optimizatoin Network

    python train_stage_3

Test：

    to test Motion Part Proposal Module and Motion Attribute Proposal Module

    python test_stage_12 --stage=1

    to test a Proposal Matching Module

    python test_stage_12 --stage=2 --batch_size=32

    to test a Motion Optimizatoin Network

    python test_stage_3

Other:
    nms.m: This file is used for merging proposal.
    generate_stage_2_train_data.m: This file is used for generating stage 2 training_data.
    generate_stage_3_train_data.m: This file is used for generating stage 3 training_data.
