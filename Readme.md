## Readme  

> C : 训练样本为一维 CBF 估计数据   
> R : 训练样本为二维 R 矩阵估计数据

### 1. generate training samples 

    cbf/music_doa.m are functions for DOA estimation;

    train_geneRnC.m is a script for generating on_grid train_set.mat;  

    train_offgrid.m is a script for generating off_grid trainoff_set.mat;


### 2. CNN model training
>**TensorFlow** needs to be installed

    train_R.py is a script for CNN_ongrid training;  

    train_offgrid.py is a script for CNN_offgrid training;

    figLoss.py plots loss curve;

With the models available in the folder, skip ahead to  **step 3**
    
    
### 3. One test

    onetest.m generate onetest smaple;
    so is onetest_offgrid.m ;

    Onetest_R.py predicts the result of CNN_ongrid and feeds back cnn_predict_OneTestR.mat；

    onetest_offgrid.py feeds back cnn_predict_OneTest_offgrid.mat；

    OneTestPerformance.m shows the performance of music/cnn_doa

    perform_offgrid.m shows the DOA spectrum of offgrid estimation


### 4. Comprehensive test

    test_gene_interval.m generates equally spaced signal samples;

    test_interval.py estimates the equally spaced signal DOA ;

    preform_interval.m plots the error scatter;


    test_geneRnC.m generates samples with the change of variables(SNR/kelm/……);
    
    test_R.py estimates the DOA with CNN and feed back cnn_predict.mat；

    AccuracyRMSE.m compares the performance of CNN and classic method;


### 5. some functions

    bce_function has the regularization function for offgrid model training;

    getPeak.m seeks the DOA of incident signals;
    
    ShotOrNot.m counts accuracy and RMSE of Estimation;

![概览](https://github.com/wotwoker/Off-grid_signal_DOA_estimation_based_on_CNN/blob/main/pic_output/CNN%E6%A6%82%E8%A7%88.png?raw=true)

![效果图](https://raw.githubusercontent.com/wotwoker/Off-grid_signal_DOA_estimation_based_on_CNN/102f588718a48322ce39349fc41733ef9fac5d2c/pic_output/offgrid2%E4%BF%A1%E5%8F%B7%E5%AE%8C%E7%BE%8E.svg)




