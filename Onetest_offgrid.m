% 生成单个测试样本：OneTestSet_offgrid.mat
% thetaOneTest_on（样本的入射角，整数）: {1,2,3,4} x 1 double
% Signal_eta_on (训练输入特征，整数得的R）: 8 x 8 x 2 double
% Signal_label_on（训练标签，01标签）: 1 x 181 double
% thetaOneTest_off（样本的入射角，包含小数）: {1,2,3,4} x 1 double
% Signal_eta_off (训练输入特征，有小数得的R）: 8 x 8 x 2 double
% Signal_label_off（训练标签，小数标签）: 1 x 181 double

clc; clear; close all; warning off
%%模型基本参数
dd = 0.5;               % 阵元间距波长比
snr = 10;               % input SNR (dB)
kelm = 8;               % 阵元数=8
snapshot = 512;         % 快拍数
phi_start = -90;        % 定义角区间起点
phi_end = 90;           % 定义角区间终点
Phi = phi_start:phi_end; % 定义叫区间
P = length(Phi);         % 定义角度数=180
kelmArr = (0:kelm-1)+0*randn(1,kelm);

%% 产生theta_train
%thetaOneTest_off = [-31.5837 -8.2913]       %信号角度（含小数）  
thetaOneTest_off = [-60 + 120 * rand,-60 + 120 * rand]
thetaOneTest_on = round(thetaOneTest_off);  %信号角度（整数）
numSignal = length(thetaOneTest_off);   % 信号源数

%% 产生Signal_eta_off和Signal_label_off（小数标签）
Signal_eta_off = zeros(kelm,kelm,2*1);
Signal_eta_on = zeros(kelm,kelm,2*1);
Signal_label_off = zeros(1,P)-1;
Signal_label_on = zeros(1,P);
Signal = randn(numSignal,snapshot);
A_off = exp(-1j*2*pi*kelmArr'*dd*sind(thetaOneTest_off));% 导向矩阵
A_on = exp(-1j*2*pi*kelmArr'*dd*sind(thetaOneTest_on));
X_off = A_off*Signal;
X_on = A_on*Signal;
X1_off = awgn(X_off,snr,'measured');    
X1_on = awgn(X_on,snr,'measured');  
R_off=1/snapshot*(X1_off*X1_off');    %协方差矩阵（2维复数）
R_on=1/snapshot*(X1_on*X1_on');    %协方差矩阵（2维复数）
normR_off = norm(R_off);  % 计算R的范数（模长）
normR_on = norm(R_on);  
Signal_eta_off(:,:,1) = real(R_off) / normR_off;  % 保存CNN_R的单样本测试集特征
Signal_eta_off(:,:,2) = imag(R_off) / normR_off;  
Signal_eta_on(:,:,1) = real(R_on) / normR_on;  % 保存CNN_R的单样本测试集特征
Signal_eta_on(:,:,2) = imag(R_on) / normR_on; 
Signal_label_off(round(thetaOneTest_off)+91) = thetaOneTest_off-round(thetaOneTest_off);
Signal_label_on(round(thetaOneTest_on)+91) = ones(1,length(thetaOneTest_on));

%% 保存数据
save('OneTestSet_offgrid.mat','thetaOneTest_off','Signal_eta_off','Signal_label_off'...
     ,'thetaOneTest_on','Signal_eta_on','Signal_label_on','Phi');

