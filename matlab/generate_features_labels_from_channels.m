%% This script convert csi channels to features and labels for Dloc training
% Tuneable Parameters
GRID_SIZE = 0.1;     % the output grid size of each pixel
OUTPUT_SIGMA = 0.25; % the gaussian variance of the output gaussian target

%% dataset setting
% change this for each dataset
data_path = "/media/ehdd_8t1/chenfeng/csi_data/dloc_pc2_10-3-2020/analysis/results-dloc_pc2_10-3-2020-comp=1.mat";
save_dir = "/media/ehdd_8t1/chenfeng/DLoc_data";

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

% for tx_idx=1:n_tx
tx_idx = 4;
ant_range = index(tx_idx,:); % index corresponding to data from tx_idx th tx antenna
channels{tx_idx} = channels3_4D(:,:,:,ant_range);
d_pred_cell{tx_idx} = d_pred(:,ant_range);
% end

% only process 1 tx at a time
channels = cat(1,channels{:});          % [n_point*n_tx, n_sub, n_rx, n_ap]
d_pred = cat(1, d_pred_cell{:});        % [n_point*n_tx, n_ap]
labels = repmat(robot_xy, [1 1]);    % [n_point*n_tx, 2->(x,y)]
real_tof = repmat(real_tof, [1 1]);  % [n_point*n_tx, n_ap]

%% Display shape matrix
fprintf('number of transmitter antenna: %d\n',n_tx)
fprintf('processing tx: %d\n',tx_idx)
fprintf('size(channels3_4D): %s \n', mat2str(size(channels3_4D)))
fprintf('size(channels): %s \n', mat2str(size(channels)))
fprintf('size(d_pred): %s \n', mat2str(size(d_pred)))
fprintf('size(labels): %s \n', mat2str(size(labels)))
fprintf('size(real_tof): %s \n', mat2str(size(real_tof)))

%% create x, y grid points. Expand space to be 2 times in both x and y
x_width = max(labels(:,1)) - min(labels(:,1));
y_width = max(labels(:,2)) - min(labels(:,2));

x_min_new = min(labels(:,1)) - 0.5 * x_width;
x_max_new = max(labels(:,1)) + 0.5 * x_width;

y_min_new = min(labels(:,2)) - 0.5 * y_width;
y_max_new = max(labels(:,2)) + 0.5 * y_width;

x_values = x_min_new:GRID_SIZE:x_max_new; % x axis grid points
y_values = y_min_new:GRID_SIZE:y_max_new; % y axis grid points

fprintf('original (x_min, x_max): (%f,%f), new (x_min, x_max): (%f,%f)\n',...
    min(labels(:,1)),...
    max(labels(:,1)),...
    x_min_new,...
    x_max_new)
fprintf('original (y_min, y_max): (%f,%f), new (y_min, y_max): (%f,%f)\n',...
    min(labels(:,2)),...
    max(labels(:,2)),...
    y_min_new,...
    y_max_new)

%% user check
m = input('Do you want to continue? y/n:','s');
if m=='n' || m=='N'
    disp('Process terminated')
    exit
end

%% AoA estimation + TOF compensation
[n_points,n_sub,n_ant,n_ap] = size(channels);

% Estimating AoA
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

% Real ToF compensation
channels_wo_offset = zeros(size(channels));
for i=1:n_points
    for j=1:n_ap
        channels_wo_offset(i,:,:,j) = squeeze(channels(i,:,:,j)).*...
            exp(1j*2*pi*opt.freq' * (d_pred(i,j) - real_tof(i,j)) / 3e8);
    end

    if(mod(i,1000)==0)
        fprintf('channels_wo_offset, sample %d\n',i);
    end
end

%% Get features
tic
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
    
    if(mod(i,100)==0)
        fprintf('Generating features, sample %d\n',i);
    end
end
toc

%% create ground truth label for training
labels_gaussian_2d = get_gaussian_labels(labels,...
    OUTPUT_SIGMA,...
    x_values,...
    y_values);

% save all data
dataset_train_name = sprintf('dataset_%s_all_tx%d.mat',dataset_name,tx_idx);
save(fullfile(save_dir, dataset_train_name),...
    'features_w_offset',...
    'features_wo_offset',...
    'labels_gaussian_2d',...
    'labels',...
    'x_values',...
    'y_values',...
    'dataset_name',...
    '-v7.3');
fprintf('data saved in %s\n', fullfile(save_dir, dataset_train_name))