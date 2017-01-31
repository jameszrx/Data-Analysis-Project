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

%problem 4_2
for k1=10
    N = size(item_id);
    N = N(1);
    index = randperm(N)';
    threshold = 0:0.1:9;
    precision = zeros(1, length(threshold));
    recall = zeros(1, length(threshold));
    for i = 0:9
        R_i = R;
        index_i = index(i*(N/10)+1 : (i+1)*(N/10));
        for j = 1:size(index_i)
            R_i(user_id(index_i(j)), item_id(index_i(j))) = 0;
        end
        [U_i, V_i] = wnmfrule1(R_i, k1);
        R_i_predict = U_i*V_i;
        R_i_predict = R_i_predict.*R;
    
        prec = zeros(1, length(threshold));
        reca = zeros(1, length(threshold));
        k = 1;
        for thres = threshold
            tp = 0;
            fp = 0;
            fn = 0;
            for j = 1:size(index_i)
                if R_i_predict(user_id(index_i(j)), item_id(index_i(j))) >= thres && R(user_id(index_i(j)), item_id(index_i(j))) >= 4
                    tp = tp + 1;
                elseif R_i_predict(user_id(index_i(j)), item_id(index_i(j))) >= thres && R(user_id(index_i(j)), item_id(index_i(j))) <= 3
                    fp = fp + 1;
                elseif R_i_predict(user_id(index_i(j)), item_id(index_i(j))) < thres && R(user_id(index_i(j)), item_id(index_i(j))) >= 4
                    fn = fn + 1;
                end
            end
            prec(k) = tp/(tp + fp);
            reca(k) = tp/(tp + fn);
            k = k + 1;
        end
    
        precision = precision + prec;
        recall = recall + reca;
    end

    precision = precision / 10;
    recall = recall / 10;

%     if k1 == 10
        figure(1);
        plot(recall, precision, 'b-', 'LineWidth', 2);
        title('ROC curve when \lambda = 1');
        xlabel('recall');
        ylabel('precision');
        hold on;
%     elseif k1 == 50
%         plot(recall, precision, 'r-', 'LineWidth', 2);
%         hold on;
%     else
%         plot(recall, precision, 'g-', 'LineWidth', 2);
%     end
end
