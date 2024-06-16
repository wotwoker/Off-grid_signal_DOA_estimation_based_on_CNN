%绘制离网格的预测曲线
clear;clc;close all;
load('cnn_predict_OneTestR.mat','P_cnn_R'); % 预测的整数信息
load('cnn_predict_OneTest_offgrid.mat','P_cnn_offgrid','P_cnn_offgrid_tanh'); % 预测的小数信息
load('OneTestSet_offgrid.mat','thetaOneTest_off','Phi');

% %预测值（结合整数与小数）
% pj=find(P_cnn_R > 0.5); %峰值类别（整数）
% zj=P_cnn_offgrid(pj); %峰值幅度（小数）
% result=-1*ones(1,181);
% result(pj)=zj;
%真实值
thetaOneTest_off
Signal_eta_int = round(thetaOneTest_off);
Signal_eta_dec = thetaOneTest_off - round(thetaOneTest_off);
%% 预测值修正（针对两个信源）
%pj=find(P_cnn_R > 0.8);%峰值类别（整数）
pj=getPeak(double(P_cnn_offgrid_tanh),2)+91;%两种找类别的方法
theta_rec = zeros(1,length(pj));
th = -0.6; %选择合适的修正阈值           %%改成差值的阈值
for i = 1:length(pj)
    p1=pj(i); %整数角度索引值
    p2=p1+1;
    %p3=p1-1;
    z1=P_cnn_offgrid_tanh(p1);
    z2=P_cnn_offgrid_tanh(p2);
    %z3=P_cnn_offgrid_tanh(p3);
    if z1 > th && z2 > th 
        theta_rec(i)=((p1-91)*(z1+1)+(p2-91)*(z2+1))/(z1+z2+2);
    %elseif abs(z1 - z3) > th
    %    theta_rec(i)=((p1-91)*(z1+1)+(p3-91)*(z3+1))/(z1+z3+2);
    %elseif abs(z1 - z2) > th && abs(z1 - z3) > th
    %    theta_rec(i)=((p1-91)*(z1+1)+(p3-91)*(z3+1)+(p2-91)*(z2+1))/(z1+z2+z3+3);
    else
        theta_rec(i)=p1-91+z1;
    end
end
theta_rec

%% 绘图
figure(1);
% set(gcf, 'Position', [100,400,1200, 400]);
% subplot(1,2,1);% 结合了整数与小数模型
% plot(Phi,result,'-','Linewidth',1.2);hold on 
% plot(Signal_eta_int,Signal_eta_dec,'*','Linewidth',1.2);hold on;
% % 对于每个点，画出与坐标轴的虚线
% for i = 1:length(Signal_eta_int)% 垂直虚线 % 水平虚线
%     line([Signal_eta_int(i), Signal_eta_int(i)], [-1, Signal_eta_dec(i)], 'LineStyle', '--', 'Color', 'k');
%     line([-90, Signal_eta_int(i)], [Signal_eta_dec(i), Signal_eta_dec(i)], 'LineStyle', '--', 'Color', 'k');
% end
% xlim([-90, 90]);%ylim([-1.01, 0.6]);
% legend('预测值','真实值');title('整数与小数结合');grid off;
% 
% subplot(1,2,2);% tanh激活后输出
plot(Phi,P_cnn_offgrid_tanh,'-','Linewidth',1.2);hold on;
plot(Signal_eta_int,Signal_eta_dec,'*','Linewidth',1.2);hold on;
for i = 1:length(Signal_eta_int)% 垂直虚线 % 水平虚线
    line([Signal_eta_int(i), Signal_eta_int(i)], [-1, Signal_eta_dec(i)], 'LineStyle', '--', 'Color', 'k');
    line([-90, Signal_eta_int(i)], [Signal_eta_dec(i), Signal_eta_dec(i)], 'LineStyle', '--', 'Color', 'k');
end
xlim([-90, 90]);ylim([-1.01, 0.6]);
xlabel('网络输出的整数信息(°)');ylabel('网络输出的小数信息(°)');
legend('预测值','真实值');title('离格激活输出');grid on;

figure(2);
plot(Phi,P_cnn_offgrid,'-','Linewidth',1.2);hold on;%只有offgrid模型
plot(Signal_eta_int,Signal_eta_dec,'*','Linewidth',1.2);hold on;
xlim([-90, 90]);
xlabel('网络输出的整数信息(°)');ylabel('网络的中间输出z');
legend('预测值','真实值');title('离格直接输出');grid on;


