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

%problem 1

[U1, V1] = wnmfrule(R, 10);
error_m1 = W.*(R - (U1*V1)).^2;
error_1 = sum(error_m1(:));
fprintf('For k = 10, the total least squared error is: %f \n', error_1);

[U2, V2] = wnmfrule(R, 50);
error_m2 = W.*(R - (U2*V2)).^2;
error_2 = sum(error_m2(:));
fprintf('For k = 50, the total least squared error is %f\n', error_2);

[U3, V3] = wnmfrule(R, 100);
error_m3 = W.*(R - (U3*V3)).^2;
error_3 = sum(error_m3(:));
fprintf('For k = 100, the total least squared error is: %f\n', error_3);