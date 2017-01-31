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


%problem 2

N = size(item_id);
N = N(1);
index = randperm(N)';

error_avg_abs = zeros(1, 10);

for i = 0:9
    R_i = R;
    index_i = index(i*(N/10)+1 : (i+1)*(N/10));
    for j = 1:size(index_i)
        R_i(user_id(index_i(j)), item_id(index_i(j))) = 0;
    end
    [U_i, V_i] = wnmfrule(R_i, 100);
    R_i_predict = U_i*V_i;
    err_abs = 0;
    for j = 1:size(index_i)
        err_abs = err_abs + abs(R(user_id(index_i(j)), item_id(index_i(j))) - R_i_predict(user_id(index_i(j)), item_id(index_i(j))));
    end
    error_avg_abs(i+1) = err_abs/(N/10);
end

