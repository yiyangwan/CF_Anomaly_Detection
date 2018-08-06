clear
filePath = 'C:\Users\SQwan\Documents\MATLAB\CF\dataset\'; % dataset location
data_table = readtable('Clean_data.csv');
v        = table2array(data_table(12001:16000,2));
delta_t = 0.1;

x = zeros(size(v));

x(1) = v(1) * delta_t;
for i = 2:length(v)
    x(i) = x(i-1) + v(i) * delta_t;
end

s_train = [x,v]; 

save(strcat(filePath,'rawdata.mat'),'s_train');