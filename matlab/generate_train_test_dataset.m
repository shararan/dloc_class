%% 
data_path = '';
test_bbox = {}; % bounding box that selects test data. {[x_min,x_max,y_min,y_max], ...}

%% train / test split
dataset = load(data_path);
[train_idxs, test_idxs] = split_train_test(dataset.labels, test_bbox);

% features_w_offset
features_w_offset_train = dataset.features_w_offset(train_idxs,:,:,:);
features_w_offset_test = dataset.features_w_offset(test_idxs,:,:,:);

% features_wo_offset
features_wo_offset_train = dataset.features_wo_offset(train_idxs,:,:,:);
features_wo_offset_test = dataset.features_wo_offset(test_idxs,:,:,:);

% labels_gaussian_2d
labels_gaussian_2d_train = dataset.labels_gaussian_2d(train_idxs,:,:);
labels_gaussian_2d_test = dataset.labels_gaussian_2d(test_idxs,:,:);

% labels
labels_train = dataset.labels(train_idxs,:);
labels_test = dataset.labels(test_idxs,:);

% x and y axis range
x_values = dataset.x_values;
y_values = dataset.y_values;

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
    'x_values',...
    'y_values',...
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
    'x_values',...
    'y_values',...
    '-v7.3');