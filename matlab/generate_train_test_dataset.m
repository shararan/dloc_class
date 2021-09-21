%% 
data_path = '/media/ehdd_8t1/chenfeng/DLoc_data/dloc_pc2_10-3-2020/dataset_dloc_pc2_10-3-2020_all_tx1.mat'; % mat file created by generate_features_labels_from_channels.m
test_bbox = {[-16.5,-5,3,6.53],...
             [-5,7.4,3,6.53],...
             [-16.5,-5,-0.16,3],...
             [-5,7.4,-0.16,3]}; % bounding box that selects test data. {[x_min,x_max,y_min,y_max], ...}
[save_dir,name,ext] = fileparts(data_path);

%% cross validation
dataset = load(data_path);
for i = 1:length(test_bbox)
    [train_idxs, test_idxs] = split_train_test(dataset.labels, test_bbox{i});
    fprintf('fold %d: test = %f of total data\n',i, length(test_idxs)/size(dataset.labels,1))
end
% m = input('Do you want to continue? y/n:','s');
% if m=='n' || m=='N'
%     disp('Process terminated')
%     exit
% end

%% save train/test data for each fold
for i = 1:length(test_bbox)
    [train_idxs, test_idxs] = split_train_test(dataset.labels, test_bbox{i});

    % save train
    train_name = sprintf('%s_train_fold_%d.mat', dataset.dataset_name,i);
    save_dataset(dataset, train_idxs, train_name, save_dir);

    % save test
    test_name = sprintf('%s_test_fold_%d.mat', dataset.dataset_name,i);
    save_dataset(dataset, test_idxs, test_name, save_dir);
    fprintf('fold %d train/test is saved.\n',i)
end