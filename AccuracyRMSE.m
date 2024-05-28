% 分析准确率和均方根误差
% 将CNN的预测结果与 MUSIC 作比较
clear;clc;close all
load('test_set_snr.mat','theta_test','VariableARR','Signal_eta','Signal_eta_forC','estMUSIC');
nLevel = size(estMUSIC,1);     % 分集的个数（snr的种类/阵列元个数的种类等）
nSignal = size(estMUSIC,2);    % 信源个数
nsample = size(estMUSIC,3);    % 每个集的样本个数
threshold = 1;                  % 认为估计正确的门限

%% 估计准确率与均方根误差
[probMUSIC,rmseMUSIC] = ShotOrNot(estMUSIC,theta_test,threshold);  

%% 加载CNN的预测结果
load('cnn_predict_C.mat','estCNN_C');
load('cnn_predict_R.mat','estCNN_R');
estCnn_C = zeros(nLevel,nSignal,nsample);
estCnn_R = zeros(nLevel,nSignal,nsample);
for iLevel = 1:nLevel
    for iSample = 1:nsample
        cnn_doa_C = reshape(estCNN_C(iLevel,iSample,:),181,1);
        estCnn_C(iLevel,:,iSample) = getPeak(cnn_doa_C,2);     % CNN_C特征谱的谱峰搜索
        
        cnn_doa_R = reshape(estCNN_R(iLevel,iSample,:),181,1);
        estCnn_R(iLevel,:,iSample) = getPeak(cnn_doa_R,2);     % CNN_R特征谱的谱峰搜索
    end
end
[probCNN_C,rmseCNN_C] = ShotOrNot(estCnn_C,theta_test,threshold); 
[probCNN_R,rmseCNN_R] = ShotOrNot(estCnn_R,theta_test,threshold); 

Linewidth = 1.5;
figure;
plot(VariableARR,probMUSIC,'--','Linewidth',Linewidth);hold on
plot(VariableARR,probCNN_C,'.-','Linewidth',Linewidth);hold on
plot(VariableARR,probCNN_R,':','Linewidth',Linewidth);hold off
xlabel('SNR(dB)');ylabel('DOA估计准确率')
legend('MUSIC','CNN_C','CNN_R');
title('Change of Number of SNR');

figure;
plot(VariableARR,rmseMUSIC,'--','Linewidth',Linewidth);hold on
plot(VariableARR,rmseCNN_C,'.-','Linewidth',Linewidth);hold on
plot(VariableARR,rmseCNN_R,':','Linewidth',Linewidth);hold off;

xlabel('SNR(dB)');ylabel('RMSE(°)');%xlabel('快拍数');
legend('MUSIC','CNN_C','CNN_R');
title('Change of Number of SNR');

