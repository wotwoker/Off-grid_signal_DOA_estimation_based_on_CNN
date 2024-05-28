% 将CNN的预测结果与/MUSIC/ESPRIT作比较

clear;clc;close all;
load('cnn_predict_OneTestC.mat','P_cnn_C');
load('cnn_predict_OneTestR.mat','P_cnn_R');
load('cnn_predict_OneTest_offgrid.mat','P_cnn_offgrid');
load('OneTestSet.mat','thetaOneTest','Signal_label','Signal_eta','Phi','P_MUSIC');


Linewidth = 1.5;
k=zeros(1,length(thetaOneTest))+1;
%plot(Phi,P_MUSIC,'-','Linewidth',1);hold on;
%plot(Phi,P_cnn_C,'-.','Linewidth',Linewidth);hold on;
plot(Phi,P_cnn_R,'-.','Linewidth',Linewidth);hold on;
plot(thetaOneTest,k,'*','Linewidth',Linewidth);hold on;

xlabel('角度(°)');ylabel('空间谱');xlim([-90, 90]);
%legend('MUSIC','真实角度');
%legend('1D-CNN','真实角度');
legend('2D-CNN','真实角度')
grid on;

