%% data processing function
function [x_l,v_l] = data_process(raw_data)
% data_process function extracts the leading vehicle infomation in order to
% implement car-following model
% input: raw data
% output: location, speed, and acceleration of the leading vehicle
x_l = raw_data(:,1);
v_l = raw_data(:,2);
end