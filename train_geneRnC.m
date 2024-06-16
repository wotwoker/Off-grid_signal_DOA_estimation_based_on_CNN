% 生成训练集：traning_set.mat 数据存储文件
% theta_train（训练样本的入射角）: 2 x ____ double（目标数 x 样本数）
% Signal_eta（输入特征 R的2@M x M张量重构）: 8 x 8 x ____ double （阵元数 x 阵元数 x 2倍样本数）
% Signal_eta_forC（输入特征 CBF-DOA）: ____ x 181 double （样本数 x 输入向量）
% Signal_label（训练标签）: ____ x 181 double （样本数 x 网格数）

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
    theta1 = [theta1,randi([-60,60])]; %追加数据共nsample个
    theta2 = [theta2,randi([-60,60])];
end
theta_train = [theta1;theta2];

%% 产生Signal_eta和Signal_label
Signal_eta = zeros(kelm,kelm,2*nsample);
Signal_eta_forC = zeros(length(theta_train),P);
Signal_label = zeros(length(theta_train),P);
for iThetaTrain = 1:length(theta_train)      % 对于每个训练样本
    S0 = randn(numSignal,snapshot);     %模拟出信号源
    A = exp(-1j*2*pi*(0:kelm-1)'*dd*sind(theta_train(:,iThetaTrain)')); % 导向矩阵
    X = A*S0;
    X1 = awgn(X,randi([-10,20]),'measured');    % 加[-10:20]dB的噪声  
    R=1/snapshot*(X1*X1');    %协方差矩阵（2维复数）
    normR = norm(R);  % 计算R的范数（模长）
    Signal_eta(:,:,1+2*(iThetaTrain-1)) = real(R) / normR;  % 将归一化后的实部赋值
    Signal_eta(:,:,2+2*(iThetaTrain-1)) = imag(R) / normR;  % 将归一化后的虚部赋值
    Signal_eta_forC(iThetaTrain,:) = cbf_doa(X1,numSignal,dd,Phi);  % 做 CBF-DOA估计
    Signal_label(iThetaTrain,round(theta_train(:,iThetaTrain))+91) = ones(1,numSignal);   % 入射信号标签为 1 
end

%% 保存训练集合
save('train_set.mat','theta_train','Signal_label','Signal_eta','Signal_eta_forC');

%% 查看样本
load('train_set.mat','theta_train','Signal_label','Signal_eta','Signal_eta_forC');
iSample = floor(rand * length(theta_train));    %随机选择样本用于可查看
figure('Position', [200,100,900, 450]);  % 查看某样本R的实虚部图形
subplot(1, 2, 1);  % 显示实部
imagesc(Signal_eta(:,:,1+2*(iSample-1)));  
title('Real Part of R'); colorbar; axis square;  %标题；色块；方形显示
subplot(1, 2, 2);  % 显示虚部
imagesc(Signal_eta(:,:,2+2*(iSample-1)));  
title('Imaginary Part of R'); colorbar; axis square;  

figure();
subplot(1, 2, 1); 
plot(Phi,Signal_eta_forC(iSample,:),'LineWidth',1.5);hold on;
stem(find(Signal_label(iSample,:)==1)-91,ones(1,numSignal),'LineWidth',1)
grid on;xlim([-90,90]); ylim([-0.1,1.1]);xlabel('角度(°)');
legend('CBF预测','真实角度');hold off;
subplot(1, 2, 2); 
plot(Phi,Signal_eta_forC(iSample+1,:),'LineWidth',1.5);hold on;
stem(find(Signal_label(iSample+1,:)==1)-91,ones(1,numSignal),'LineWidth',1)
grid on;xlim([-90,90]); ylim([-0.1,1.1]);xlabel('角度(°)');
legend('CBF预测','真实角度');hold off;
