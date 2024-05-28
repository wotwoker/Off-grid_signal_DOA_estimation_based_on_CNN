%绘制等间隔信号的误差分析曲线
clear;clc;close all;
load('cnn_predict_ITVR.mat','estCNN_R'); % 2D-CNN的估计谱
load('cnn_predict_ITVC.mat','estCNN_C'); % 1D-CNN的估计谱
load('cnn_predict_ITVoff.mat','estCNN_off'); % 1D-CNN的估计谱
load('test_set_interval.mat','theta_test');

nSignal = size(theta_test,1);    % 信源个数
nsample = size(theta_test,2);    % 每个集的样本个数
estCnn_C = zeros(nSignal,nsample);
estCnn_R = zeros(nSignal,nsample);

estCnn_on = zeros(nSignal,nsample);
estCnn_off = zeros(nSignal,nsample);
for iSample = 1:nsample
%     cnn_doa_C = reshape(estCNN_C(iSample,:),181,1);
%     estCnn_C(:,iSample) = getPeak(cnn_doa_C,2);     % CNN_C特征谱的谱峰搜索

    %cnn_doa_R = reshape(estCNN_R(iSample,:),181,1);
    %estCnn_R(:,iSample) = getPeak(cnn_doa_R,2);     % CNN_R特征谱的谱峰搜索
    
    cnn_doa_on = reshape(estCNN_R(iSample,:),181,1);
    estCnn_on(:,iSample) =getPeak(cnn_doa_on,2);
    theta_rec = zeros(1,length(estCnn_on(:,iSample)));
    th=-0.6;
    for i = 1:length(estCnn_on(:,iSample))
        p1=estCnn_on(i,iSample); %整数角度索引值，
        p2=p1+1;
        z1=estCNN_off(iSample,p1+91);
        z2=estCNN_off(iSample,p2+91);
        if z1 > th && z2 > th 
            theta_rec(i)=((p1)*(z1+1)+(p2)*(z2+1))/(z1+z2+2);
        else
            theta_rec(i)=z1+p1;
        end
    end
    estCnn_off(:,iSample) = theta_rec; %CNN_off特征谱的全部信息
    
end
estCnn_C = sort(estCnn_C, 1, 'ascend'); %对列重新排序
%estCnn_R = sort(estCnn_R, 1, 'ascend');
estCnn_offgrid = sort(estCnn_off, 1, 'ascend');%离网格信号的全部信息


column = 1:nsample;
% 绘制散点图
% figure; 
% subplot(1,2,1);
% scatter(column, estCnn_R(1, :),'*');hold on;% 绘制第一行的数据点
% scatter(column, estCnn_R(2, :),'*');hold on; % 绘制第二行的数据点
% line([100,100], [-60,60]);
% line([200,200], [-60,60]);
% line([300,300], [-60,60]);
% xlabel('样本索引');ylabel('DOA估计(°)');ylim([-60,60]);
% subplot(1,2,2);
% scatter(column, estCnn_R(1, :)-theta_test(1, :),'*');hold on;
% scatter(column, estCnn_R(2, :)-theta_test(2, :),'*');hold on;
% xlabel('样本索引');ylabel('DOA误差(°)');ylim([-5,5]);

%绘制离网格估计散点图
figure; 
subplot(1,2,1);
scatter(column, estCnn_offgrid(1, :),'*');hold on;% 绘制第一行的数据点
scatter(column, estCnn_offgrid(2, :),'*');hold on; % 绘制第二行的数据点
line([100,100], [-60,60]);
line([200,200], [-60,60]);
line([300,300], [-60,60]);
legend('第一个信号','第二个信号');grid on;
xlabel('样本索引');ylabel('DOA估计(°)');ylim([-60,60]);

subplot(1,2,2);
scatter(column, (estCnn_offgrid(1, :)-theta_test(1, :)),'*');hold on;
scatter(column, (estCnn_offgrid(2, :)-theta_test(2, :)),'*');hold on;
legend('第一个信号误差','第二个信号误差');grid on;
xlabel('样本索引');ylabel('DOA误差(°)');ylim([-1,1]);





