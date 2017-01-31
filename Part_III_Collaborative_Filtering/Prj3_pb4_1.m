clear;clc;

%load the data
data_in = importdata('u.data');
user_id = data_in(:, 1);
item_id = data_in(:, 2);
rating = data_in(:, 3);

row_num = max(user_id);
col_num = max(item_id);

R = zeros(row_num, col_num);
W = zeros(row_num, col_num);

for i = 1:size(user_id)
    R(user_id(i), item_id(i)) = rating(i);
    W(user_id(i), item_id(i)) = 1;
end

%problem 4


%residual: 1.4920e+00   lambda=0
%residual: 1.2007e+00   lambda=0.01
%residual: 3.3354e+00   lambda=0.1
%residual: 1.5541e+01   lambda=1
[U1, V1] = wnmfrule1(R, 10);
%error_m1 = W.*(R - (U1*V1)).^2;
%error_1 = sum(error_m1(:));
%fprintf('For k = 10, the total least squared error is: %f \n', error_1);

%residual: 3.0943e+00   lambda=0
%residual: 3.2501e+00   lambda=0.01
%residual: 3.5110e+00   lambda=0.1
%residual: 1.4417e+01   lambda=1
[U2, V2] = wnmfrule1(R, 50);
%error_m2 = W.*(R - (U2*V2)).^2;
%error_2 = sum(error_m2(:));
%fprintf('For k = 50, the total least squared error is %f\n', error_2);

%residual: 4.7819e+00   lambda=0
%residual: 4.6323e+00   lambda=0.01
%residual: 4.6233e+00   lambda=0.1
%residual: 1.4163e+01   lambda=1
[U3, V3] = wnmfrule1(R, 100);
%error_m3 = W.*(R - (U3*V3)).^2;
%error_3 = sum(error_m3(:));
%fprintf('For k = 100, the total least squared error is: %f\n', error_3);
