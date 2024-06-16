% 生成变量（信噪比/阵列个数/……）分集的测试集：test_set.mat
% 
% theta_test（训练样本的入射角）: {1,2,3} x sample double（目标数 x 样本数）
% VariableARR (分集变量 信噪比/阵列个数/……） 2:4:32
% Signal_eta（训练输入特征）: variable x 8 x 8 x ____ double （阵元数 x 阵元数 x 2倍样本数）
% Signal_eta_forC（训练输入特征）: sample x 181 double（样本数 x 输入向量）
% estMUSIC（MUSIC算法检测的入射角）: variable x 2 x sample double （分集 x 目标数 x 样本数）

clc; clear; close all;
dd = 0.5;               % space 
numSignal = 2;          % number of DOA
phi_start = -90;        % 定义角区间起点
phi_end = 90;           % 定义角区间终点
Phi = phi_start:1:phi_end; % 定义角区间
P = length(Phi);        % 定义角度数=180

%% 设置测试信号基本参数及默认参数
snr = 0;         % 默认信噪比
sample = 300;    % 产生n个测试样本:建议1000-5000
kelm = 8;         % 默认阵列数量
snapshot = 512;     % 默认快拍数量
theta_test = zeros(numSignal, sample); % 用于保存入射角度

%% 随机产生两个角度
for iTheta = 1:sample
%     theta1 = -60+120*rand;
%     theta2 = -60+120*rand;
%     theta_test(:,iTheta) = [theta1;theta2];
    zeta = -0.1 + rand()/5; %生成一个在【-0.1,0.1】范围内随机取值的zeta
    theta_test(:,iTheta) = [15.7+zeta;59.2+zeta]; %蒙特卡洛实验
end

%% 产生空间谱并保存
VariableARR = -10:2:20; %SNR 
%VariableARR = 5:50:555; %快拍数
Signal_eta = zeros(length(VariableARR),kelm,kelm,2*sample);
Signal_eta_forC = zeros(length(VariableARR),sample,P);
estMUSIC = zeros(length(VariableARR),numSignal,sample);

iVariable = 0;
totalTime = 0; % 初始化总时间变量
%for kelm = VariableARR %更换变量需改变量名 for kelm要改的太多了
for snr = VariableARR %更换变量需改变量名 
%for snapshot= VariableARR  % 对每一个信噪比都生成sample个测试数据
    iVariable = iVariable+1;
    for iSample = 1:sample     
        thetaOneTest = theta_test(:,iSample)';
        Signal = randn(numSignal,snapshot);
        A = exp(-1j*2*pi*(0:kelm-1)'*dd*sind(thetaOneTest));% 导向矩阵
        X = A*Signal;
        X1 = awgn(X,snr,'measured'); 
        R=1/snapshot*(X1*X1');    %协方差矩阵（2维复数）
        normR = norm(R);  % 计算R的范数（模长）
        
        Signal_eta(iVariable,:,:,1+2*(iSample-1)) = real(R) / normR;  % 保存CNN_R的测试集特征 
        Signal_eta(iVariable,:,:,2+2*(iSample-1)) = imag(R) / normR;  
        tic; % 开始计时music_doa
        P_MUSIC = music_doa(X1,numSignal,dd,Phi);    % MUSIC_DOA
        elapsedTime = toc; % 结束计时并获取时间
        totalTime = totalTime + elapsedTime; % 累加每次调用的时间
        P_CBF = cbf_doa(X1,numSignal,dd,Phi);  % CBF_DOA
        Signal_eta_forC(iVariable,iSample,:) =P_CBF; % 保存CNN_C的测试集特征   
        estMUSIC(iVariable,:,iSample) = getPeak(P_MUSIC,numSignal);    
    end
end
fprintf('music_doa在循环中的总运行时间：%.2f秒。\n', totalTime)
%% 保存数据
save('test_set_snr.mat','theta_test','VariableARR','Signal_eta','Signal_eta_forC','estMUSIC');
%save('test_set_snap.mat','theta_test','VariableARR','Signal_eta','Signal_eta_forC','estMUSIC');

