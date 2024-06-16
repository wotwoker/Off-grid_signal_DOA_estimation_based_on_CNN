% 生成5、15、30等间隔的测试集：test_set.mat
% 
% theta_test（训练样本的入射角）: {1,2,3} x sample double（目标数 x 样本数）
% Signal_eta（离格输入特征）: variable x 8 x 8 x ____ double （阵元数 x 阵元数 x 2倍样本数）
% Signal_eta_forC（训练输入特征）: sample x 181 double（样本数 x 输入向量）
% Signal_eta_on（训练网格输入特征）: variable x 8 x 8 x ____ double （阵元数 x 阵元数 x 2倍样本数）

clc; clear; close all;
dd = 0.5;               % space 
numSignal = 2;          % number of DOA
phi_start = -90;        % 定义角区间起点
phi_end = 90;           % 定义角区间终点
Phi = phi_start:1:phi_end; % 定义角区间
P = length(Phi);        % 定义角度数=180

%% 设置测试信号基本参数及默认参数
snr = 0;         % 默认信噪比
sample = 300;    % 产生n个测试样本
kelm = 8;         % 默认阵列数量
snapshot = 512;     % 默认快拍数量

%% 每100个样本产生等间隔的两个角度
theta1 = [linspace(-60,55,100),linspace(-60,45,100),linspace(-60,30,100)];
theta2 = [linspace(-60,55,100)+5.5,linspace(-60,45,100)+15.6,linspace(-60,30,100)+30.7];
theta_test = [theta1;theta2];
theta_test_on = round(theta_test);  %信号角度（整数）
%% 产生空间谱并保存

Signal_eta = zeros(kelm,kelm,2*sample);
Signal_eta_on = zeros(kelm,kelm,2*sample);
Signal_eta_forC = zeros(sample,P);
%estMUSIC = zeros(numSignal,sample);

for iSample = 1:sample     
    thetaOneTest = theta_test(:,iSample)';
    thetaOneTest_on=theta_test_on(:,iSample)';%这里入射信号角度都为整数
    Signal = randn(numSignal,snapshot);
    A = exp(-1j*2*pi*(0:kelm-1)'*dd*sind(thetaOneTest));% 导向矩阵
    A_on=exp(-1j*2*pi*(0:kelm-1)'*dd*sind(thetaOneTest_on));
    X = A*Signal;
    X_on=A_on*Signal;
    X1 = awgn(X,snr,'measured'); 
    X1_on = awgn(X_on,snr,'measured'); 
    R=1/snapshot*(X1*X1');    %协方差矩阵（2维复数）
    R_on=1/snapshot*(X1_on*X1_on');
    normR = norm(R);  % 计算R的范数（模长）
    normR_on = norm(R_on);
    
    Signal_eta(:,:,1+2*(iSample-1)) = real(R) / normR;  % 保存CNN_R的测试集特征 
    Signal_eta(:,:,2+2*(iSample-1)) = imag(R) / normR;  
    Signal_eta_on(:,:,1+2*(iSample-1)) = real(R_on) / normR_on;  % 保存CNN_R的测试集特征 
    Signal_eta_on(:,:,2+2*(iSample-1)) = imag(R_on) / normR_on;  
    P_CBF = cbf_doa(X1,numSignal,dd,Phi);  % CBF_DOA
    Signal_eta_forC(iSample,:) =P_CBF; % 保存CNN_C的测试集特征   

end

%% 保存数据
save('test_set_interval.mat','theta_test','Signal_eta','Signal_eta_forC','Signal_eta_on');

