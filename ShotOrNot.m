% 计算准确率与均方根误差使用
% estMeasure：对应DOA估计的入射角集合
% ThetaTest：初始设定的入射角度
% threshold：判断正确的门限
% AccuracyEst：估计准确度
% RmseEst：估计的均方根误差
function [AccuracyEst,RmseEst] = ShotOrNot(estMeasure,ThetaTest,threshold)
    nLevel = size(estMeasure,1);
    nSignal = 2;
    nsample = size(estMeasure,3);
    AccuracyEst = zeros(1,nLevel);
    RmseEst = zeros(1,nLevel);
    error = zeros(1,nLevel);
    for iLevel = 1:nLevel
        moto=0;
        shot = zeros(nsample,1);
        ests = reshape(estMeasure(iLevel,:,:),nSignal,nsample);
        for isample = 1:nsample
            est = sort(ests(:,isample));
            label = sort(ThetaTest(:,isample));
            if abs(est(1)-label(1))<=threshold && abs(est(2)-label(2))<=threshold
                shot(isample) = 1;
            end
            if sum((est-label).^2) < 100 % 若差的平方和小于某个阈值，则认为是有效模拟，累加到RMSE总和中
                error(iLevel) = error(iLevel)+sum((est-label).^2);
                moto=moto+1;                
            end
        end        
        AccuracyEst(iLevel) = sum(shot)/nsample;
        RmseEst(iLevel) = sqrt(error(iLevel)/(nSignal*moto));
    end
end