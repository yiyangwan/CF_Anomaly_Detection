clear

filename = 'd15.mat';

load(filename)


data_len_train = 3000;
data_len_test = 800;

% locate the optimal control senario 
for i=1:20000
    if max(itstd_time_cf(i:i+data_len_train)==0)
        special_i=i;
        break
    end
end

for j = (special_i + data_len_test):20000
    if max(itstd_time_cf(j:j+data_len_test)==0)
        special_j=j;
        break
    end    
end

s_l_train = [s_pos_6(3,special_i:special_i+data_len_train);s_vel_6(3,special_i:special_i+data_len_train)]';
s_f_train = [px_sub(special_i:special_i+data_len_train);vx_sub(special_i:special_i+data_len_train)]';

s_l_test = [s_pos_6(3,special_j:special_j+data_len_test);s_vel_6(3,special_j:special_j+data_len_test)]';
s_f_test = [px_sub(special_j:special_j+data_len_test);vx_sub(special_j:special_j+data_len_test)]';

save('train_data_15.mat','s_l_train','s_f_train')
save('test_data_15.mat','s_l_test','s_f_test')