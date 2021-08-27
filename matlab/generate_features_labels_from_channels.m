%% Script that converts channels to features and labels for Dloc
% loads the dataset from DATASET_NAME and saves the datasets
clearvars

%% Tuneable Parameters
OUTPUT_GRID_SIZE = 0.1; %the output grid size of each pixel
OUTPUT_SIGMA = 0.25; % the gaussian variance of the ouput gaussian target

%% create x and y range
x_range_max = 5;   % size of the field alone x direction
y_range_max = 5;   % size of the field alone y direction
d1 = 0:OUTPUT_GRID_SIZE:x_range_max; % x_range of image in meter
d2 = 0:OUTPUT_GRID_SIZE:y_range_max; % y_range of image in meter

%% load data
% Input mat file should contain:
% dataset_name (str): name of the dataset
% channels (matrix): size = [n_point,n_sub,n_ant,n_ap], raw csi data
% channels_tof_comp (matrix): size = [n_point,n_sub,n_ant,n_ap], csi data after tof compensation
% opt (struct): constants such as subcarrier frequencies, subcarrier_indices, bandwidth, etc
% theta_vals (array): aoa search space in radians
% d_vals (array): tof search space in meter
% xy_labels (array): size = [n_points, 2], xy ground turth labels

% load(['channels_,',DATASET_NAME,'.mat']);
channels = channels3_4D(1:10,:,:,[1,3,5,7]);
labels = robot_xy(1:10, :);
[n_points,n_sub,n_ant,n_ap] = size(channels);

%% Estimating AoA
S = get_2dsteering_matrix(theta_vals,d_vals,opt);

for i=1:n_points
    [aoa_pred(i,:),d_pred(i,:)] = get_least_tofs_aoa(...
        squeeze(channels(i,:,:,:)),...
        theta_vals,...
        d_vals,...
        opt.threshold,...
        n_ap,...
        S);
    if(mod(i,1000)==0)
        disp(i)
    end
end

%% Real ToF compensation
for i=1:n_points
    parfor j=1:n_ap
        channels_wo_offset(i,:,:,j) = squeeze(channels(i,:,:,j)).*...
            exp( 1j*2*pi*opt.freq.'*( d_pred(i,j) - real_tof(i,j) )./3e8 );
    end
end

%% resize x_range and y_range
max_x = 3*d1(end)/2;
max_y = 3*d2(end)/2;
min_x = -d1(end)/2;
min_y = -d2(end)/2;
d1 = min_x:input_grid_size:max_x;
d2 = min_y:input_grid_size:max_y;

features_w_offset = zeros(n_points,n_ap,length(d2),length(d1));
features_wo_offset = zeros(n_points,n_ap,length(d2),length(d1));

%% Get features
parfor i=1:n_points
    features_with_offset(i,:,:,:) = generate_features_abs(squeeze(channels(i,:,:,:)),...
        ap,...
        theta_vals,...
        d_vals,...
        d1,...
        d2,...
        opt);
    
    features_without_offset(i,:,:,:) = generate_features_abs(squeeze(channels_wo_offset(i,:,:,:)),...
        ap,...
        theta_vals,...
        d_vals,...
        d1,...
        d2,...
        opt);
    
    if(mod(i,1000)==0)
        disp(i);
    end
end

% labels: Nx2
labels_gaussian_2d = get_gaussian_labels(labels,...
    OUTPUT_SIGMA,...
    d1,...
    d2);

save(['dataset_',DATASET_NAME,'.mat'], ...
    'features_with_offset',...
    'features_without_offset',...
    'labels_gaussian_2d',...
    'labels',...
    '-v7.3');