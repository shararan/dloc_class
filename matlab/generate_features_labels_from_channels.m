%% This script convert csi channels to features and labels for Dloc traning
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
test_bbox = {}; % bounding box that selects test data. {[x_min,x_max,y_min,y_max], ...}

%% load data
load(data_path, ...
    'channels3_4D', ...   % size = [n_point,n_sub,n_ant,n_ap], raw csi data
    'robot_xy', ...       % size = [n_points, 2], xy ground truth labels
    'real_tof', ...       % size = [n_points, n_ap], real time of flight in m
    'theta_vals',...      % aoa search space in radians
    'd_vals',...          % tof search space in m
    'opt',...             % struct, contain constants like freq, bandwidth, etc
    'ap',...              % cell, xy coordinates of antennas on all access point
    'dataset_name',...    % str, name of dataset
    'd_pred');            % tof prediction in meter

%% reformat data
channels1 = channels3_4D(:,:,:,[1,3,5,7]);
channels2 = channels3_4D(:,:,:,[2,4,6,8]);
d_pred1 = d_pred(:,[1,3,5,7]);
d_pred2 = d_pred(:,[2,4,6,8]);
channels = cat(1, channels1, channels2);
d_pred = cat(1, d_pred1, d_pred2);
labels = repmat(robot_xy, [2 1]);
real_tof = repmat(real_tof, [2 1]);

%% create variables 
d1 = 0:GRID_SIZE:x_max; % x_range of image in meter
d2 = 0:GRID_SIZE:y_max; % y_range of image in meter
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
max_x = 3*d1(end)/2;
max_y = 3*d2(end)/2;
min_x = -d1(end)/2;
min_y = -d2(end)/2;
d1 = min_x:GRID_SIZE:max_x;
d2 = min_y:GRID_SIZE:max_y;

%% Get features
features_w_offset = zeros(n_points,n_ap,length(d2),length(d1));
features_wo_offset = zeros(n_points,n_ap,length(d2),length(d1));
for i=1:n_points
    features_w_offset(i,:,:,:) = generate_features_abs(squeeze(channels(i,:,:,:)),...
        ap,...
        theta_vals,...
        d_vals,...
        d1,...
        d2,...
        opt);
    
    features_wo_offset(i,:,:,:) = generate_features_abs(squeeze(channels_wo_offset(i,:,:,:)),...
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

%% create ground truth label for training
labels_gaussian_2d = get_gaussian_labels(labels,...
    OUTPUT_SIGMA,...
    d1,...
    d2);

%% train / test split
[train_idxs, test_idxs] = split_train_test(labels, test_bbox);

% features_w_offset
features_w_offset_train = features_w_offset(train_idxs,:,:,:);
features_w_offset_test = features_w_offset(test_idxs,:,:,:);

% features_wo_offset
features_wo_offset_train = features_wo_offset(train_idxs,:,:,:);
features_wo_offset_test = features_wo_offset(test_idxs,:,:,:);

% labels_gaussian_2d
labels_gaussian_2d_train = labels_gaussian_2d(train_idxs,:,:);
labels_gaussian_2d_test = labels_gaussian_2d(test_idxs,:,:);

% labels
labels_train = labels(train_idxs,:);
labels_test = labels(test_idxs,:);

%% save all data
[save_dir,~,~] = fileparts(data_path);

% rename train data
features_w_offset = features_w_offset_train;
features_wo_offset = features_wo_offset_train;
labels_gaussian_2d = labels_gaussian_2d_train;
labels = labels_train;
index = train_idxs;

% save train data
dataset_train_name = sprintf('dataset_%s_train.mat',dataset_name);
save(fullfile(save_dir, dataset_train_name), ...
    'features_w_offset',...
    'features_wo_offset',...
    'labels_gaussian_2d',...
    'labels',...
    'index',...
    '-v7.3');

% rename test data
features_w_offset = features_w_offset_test;
features_wo_offset = features_wo_offset_test;
labels_gaussian_2d = labels_gaussian_2d_test;
labels = labels_test;
index = test_idxs;

% save test data
dataset_test_name = sprintf('dataset_%s_test.mat',dataset_name);
save(fullfile(save_dir, dataset_test_name), ...
    'features_w_offset',...
    'features_wo_offset',...
    'labels_gaussian_2d',...
    'labels',...
    'index',...
    '-v7.3');