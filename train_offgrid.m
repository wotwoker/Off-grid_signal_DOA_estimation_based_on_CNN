% 生成训练集：traningoff_set.mat 数据存储文件
% theta_train（训练样本的入射角）: 2 x ____ double（目标数 x 样本数）
% Signal_eta（输入特征 R的2@M x M张量重构）: 8 x 8 x ____ double （阵元数 x 阵元数 x 2倍样本数）
% Signal_label（小数标签 -1 或 -0.5~0.5）: ____ x 181 double （样本数 x 网格数）

%%模型基本参数
dd = 0.5;            % 阵元间距波长比
numSignal = 2;       % number of DOA           
kelm = 8;            % 阵元数 
snapshot = 512;      % 快拍数
phi_start = -90;     % 定义角区间起点
phi_end = 90;        % 定义角区间终点
Phi = phi_start:phi_end; % 遍历角度
P = length(Phi);     % 定义角度数=180
nsample = 50000;     % 样本数 5w~10w

%% 产生theta_train  
theta1=[];theta2=[];
for iTheta = 1:nsample
    theta1 = [theta1,-60 + 120 * rand]; %含小数的随机数
    theta2 = [theta2,-60 + 120 * rand];
end
theta_train = [theta1;theta2];

%% 产生Signal_eta和Signal_label
Signal_eta = zeros(kelm,kelm,2*nsample);
Signal_label = zeros(length(theta_train),P)-1;
for iThetaTrain = 1:length(theta_train)      % 对于每个训练样本
    S0 = randn(numSignal,snapshot);     %模拟出信号源
    A = exp(-1j*2*pi*(0:kelm-1)'*dd*sind(theta_train(:,iThetaTrain)')); % 导向矩阵
    X = A*S0;
    X1 = awgn(X,randi([-10,20]),'measured');    % 加[-10:20]dB的噪声  
    R=1/snapshot*X1*X1';    %协方差矩阵（2维复数）
    normR = norm(R);  % 计算R的范数（模长）
    Signal_eta(:,:,1+2*(iThetaTrain-1)) = real(R) / normR;  % 将归一化后的实部赋值
    Signal_eta(:,:,2+2*(iThetaTrain-1)) = imag(R) / normR;  % 将归一化后的虚部赋值
    Signal_label(iThetaTrain,round(theta_train(:,iThetaTrain))+91) = theta_train(:,iThetaTrain)-round(theta_train(:,iThetaTrain));   % 入射信号标签为小数值 
end

%% 保存训练集合
save('trainoff_set.mat','theta_train','Signal_label','Signal_eta');

%% 查看样本
load('trainoff_set.mat','theta_train','Signal_label','Signal_eta');
iSample = floor(rand * length(theta_train));    %随机选择样本用于可查看
figure('Position', [200,100,900, 450]);  % 查看某样本R的实虚部图形
subplot(1, 2, 1);  % 显示实部
imagesc(Signal_eta(:,:,1+2*(iSample-1)));  
title('Real Part of R'); colorbar; axis square;  %标题；色块；方形显示
subplot(1, 2, 2);  % 显示虚部
imagesc(Signal_eta(:,:,2+2*(iSample-1)));  
title('Imaginary Part of R'); colorbar; axis square;  
