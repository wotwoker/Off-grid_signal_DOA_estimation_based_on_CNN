% 生成单个测试样本：OneTestSet.mat
% thetaOneTest（样本的入射角）: {1,2,3,4} x 1 double
% Signal_eta (训练输入特征）: 8 x 8 x 2 double
% Signal_eta_forC（训练输入特征）: 1 x 181 double
% Signal_label（训练标签）: 1 x 181 double
% Phi （遍历角度）：1×181 double
% P_MUSIC (MUSIC算法的空间谱) ：1×181 double

clc; clear; close all; warning off
%%模型基本参数
dd = 0.5;               % 阵元间距波长比
snr = 0;               % input SNR (dB)
kelm = 8;               % 阵元数=8
snapshot = 512;         % 快拍数
phi_start = -90;        % 定义角区间起点
phi_end = 90;           % 定义角区间终点
Phi = phi_start:phi_end; % 定义叫区间
P = length(Phi);         % 定义角度数=180
kelmArr = (0:kelm-1)+0*randn(1,kelm);

%% 产生theta_train
thetaOneTest = [ 0  30 ] 
%thetaOneTest = [ -65 65 ]          %信号角度  
numSignal = length(thetaOneTest);   % 信号源数

%% 产生Signal_eta和Signal_label
Signal_eta = zeros(kelm,kelm,2*1);
Signal_eta_forC = zeros(1,P);
Signal_label = zeros(1,P);
Signal = randn(numSignal,snapshot);
A = exp(-1j*2*pi*kelmArr'*dd*sind(thetaOneTest));% 导向矩阵
X = A*Signal;
X1 = awgn(X,snr,'measured');    
R=1/snapshot*(X1*X1');    %协方差矩阵（2维复数）
normR = norm(R);  % 计算R的范数（模长）
Signal_eta(:,:,1) = real(R) / normR;  % 保存CNN_R的单样本测试集特征
Signal_eta(:,:,2) = imag(R) / normR;  
Signal_eta_forC = Signal_eta_forC + cbf_doa(X1,numSignal,dd,Phi);% 保存CNN_M的单样本测试集特征
Signal_label(round(thetaOneTest)+91) = ones(1,length(thetaOneTest));

%% MUSIC_DOA
P_MUSIC = music_doa(X1,numSignal,dd,Phi);    % MUSIC_DOA
%% 保存数据
save('OneTestSet.mat','thetaOneTest','Signal_eta','Signal_eta_forC','Signal_label','Phi','P_MUSIC');


% figure('Position', [200,100,900, 450]);  % 查看某样本R的实虚部图形
% subplot(1, 2, 1);  % 显示实部
% imagesc(Signal_eta(:,:,1));  
% title('Real Part of R'); colorbar; axis square;  %标题；色块；方形显示
% subplot(1, 2, 2);  % 显示虚部
% imagesc(Signal_eta(:,:,2));  
% title('Imaginary Part of R'); colorbar; axis square;  
figure();
plot(Phi,Signal_eta_forC(1,:),'LineWidth',1.5);hold on;
stem(find(Signal_label(1,:)==1)-91,ones(1,numSignal),'LineWidth',1)
grid on;xlim([-90,90]); ylim([-0.1,1.1]);xlabel('角度(°)');
legend('CBF预测','真实角度');hold off;


