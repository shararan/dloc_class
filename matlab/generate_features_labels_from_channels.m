%% This script convert csi channels to features and labels for Dloc training
% Tuneable Parameters
GRID_SIZE = 0.1;     % the output grid size of each pixel
OUTPUT_SIGMA = 0.25; % the gaussian variance of the output gaussian target
TRAIN_SPLIT = 0.8;   % percentage for train set
TEST_SPLIT = 0.2;    % percentage for test set

%% dataset setting
% change this for each dataset
data_path = "/media/ehdd_8t1/chenfeng/phone_data/results-phone_4AP-comp=1.mat";
x_max = 5;   % size of the field alone x direction
y_max = 5;   % size of the field alone y direction

%% load data
load(data_path, ...
    'channels3_4D', ...   % size = [n_point, n_sub, n_rx_ant, n_ap*n_tx], raw csi data
    'robot_xy', ...       % size = [n_points, 2], xy ground truth labels
    'real_tof', ...       % size = [n_points, n_ap], real time of flight in m
    'd_pred',...          % size = [n_points, n_ap*n_tx], tof prediction in meter
    'theta_vals',...      % aoa search space in radians
    'd_vals',...          % tof search space in m
    'opt',...             % struct, contain constants like freq, bandwidth, etc
    'ap',...              % cell, [1,n_ap] xy coordinates of antennas on all access point
    'dataset_name');      % str, name of dataset           

%% reformat data, stack different tx data on the first dim
n_tx = size(channels3_4D,4) / length(ap); % number of transmitter antenna
channels = cell(1, length(ap));
d_pred_cell = cell(1, length(ap));
index = reshape(1:size(channels3_4D,4), n_tx, []);
for i=1:n_tx
    ant_range = index(i,:);
    channels{i} = channels3_4D(:,:,:,ant_range);
    d_pred_cell{i} = d_pred(:,ant_range);
end
channels = cat(1,channels{:});          % [n_point*n_tx, n_sub, n_rx, n_ap]
d_pred = cat(1, d_pred_cell{:});        % [n_point*n_tx, n_ap]
labels = repmat(robot_xy, [n_tx 1]);    % [n_point*n_tx, 2->(x,y)]
real_tof = repmat(real_tof, [n_tx 1]);  % [n_point*n_tx, n_ap]

%% create variables 
x_values = 0:GRID_SIZE:x_max; % x axis grid points
y_values = 0:GRID_SIZE:y_max; % y axis grid points
[n_points,n_sub,n_ant,n_ap] = size(channels);

%% Estimating AoA
% S = get_2dsteering_matrix(theta_vals,d_vals,opt);
% d_pred = zeros(n_points, n_ap);
% 
% for i=1:n_points
%     [~,d_pred(i,:)] = get_least_tofs_aoa(...
%         squeeze(channels(i,:,:,:)),...
%         theta_vals,...
%         d_vals,...
%         opt.threshold,...
%         n_ap,...
%         S);
%     
%     if(mod(i,1000)==0)
%         disp(i)
%     end
% end

%% Real ToF compensation
channels_wo_offset = zeros(size(channels));
for i=1:n_points
    parfor j=1:n_ap
        channels_wo_offset(i,:,:,j) = squeeze(channels(i,:,:,j)).*...
            exp( 1j*2*pi*opt.freq.'*( d_pred(i,j) - real_tof(i,j) )./3e8 );
    end
end

%% resize x_range and y_range
max_x = 3*x_values(end)/2;
max_y = 3*y_values(end)/2;
min_x = -x_values(end)/2;
min_y = -y_values(end)/2;
x_values = min_x:GRID_SIZE:max_x;
y_values = min_y:GRID_SIZE:max_y;

%% Get features
features_w_offset = zeros(n_points,n_ap,length(y_values),length(x_values));
features_wo_offset = zeros(n_points,n_ap,length(y_values),length(x_values));
for i=1:n_points
    features_w_offset(i,:,:,:) = generate_features_abs(squeeze(channels(i,:,:,:)),...
        ap,...
        theta_vals,...
        d_vals,...
        x_values,...
        y_values,...
        opt);
    
    features_wo_offset(i,:,:,:) = generate_features_abs(squeeze(channels_wo_offset(i,:,:,:)),...
        ap,...
        theta_vals,...
        d_vals,...
        x_values,...
        y_values,...
        opt);
    
    if(mod(i,1000)==0)
        disp(i);
    end
end

%% create ground truth label for training
labels_gaussian_2d = get_gaussian_labels(labels,...
    OUTPUT_SIGMA,...
    x_values,...
    y_values);

% save all data
dataset_train_name = sprintf('dataset_%s_all.mat',dataset_name);
save(fullfile(save_dir, dataset_train_name), ...
    'features_w_offset',...
    'features_wo_offset',...
    'labels_gaussian_2d',...
    'labels',...
    'index',...
    'x_values',...
    'y_values',...
    '-v7.3');