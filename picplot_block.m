% 定义sigmoid函数
sigmoid = @(x) 1 ./ (1 + exp(-x));
% 定义sigmoid函数的导数
sigmoid_derivative = @(x) sigmoid(x) .* (1 - sigmoid(x));
% 创建一个x值的范围以绘制函数
x = -10:0.1:10;
% 绘制sigmoid函数
subplot(1,2,1); % 创建左侧的子图
plot(x, sigmoid(x), 'b', 'LineWidth', 2);
title('Sigmoid Function');
xlabel('x');
ylabel('sigmoid(x)');
grid on;
% 绘制sigmoid函数的导数
subplot(1,2,2); % 创建右侧的子图
plot(x, sigmoid_derivative(x), 'r', 'LineWidth', 2);
title('Derivative of Sigmoid Function');
xlabel('x');
ylabel('sigmoid''(x)');
grid on;
% 增加图例
legend('Sigmoid', 'Derivative');
% 调整子图间的间距
subplot(1,2,1);
pos1 = get(gca, 'Position'); % 获取当前轴的位置
pos1(3) = 0.4; % 设置宽度
set(gca, 'Position', pos1);
subplot(1,2,2);
pos2 = get(gca, 'Position'); % 获取当前轴的位置
pos2(3) = 0.4; % 设置宽度
set(gca, 'Position', pos2);