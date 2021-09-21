function save_dataset(dataset, index, file_name, save_dir)
    % dataset (struct): data saved by generate_features_labels_from_channels.m
    % index (vector): 1xN, index of data need to be saved
    % file_name (str): name of output mat file
    % save_dir (str): directory where mat file is saved

    % features_w_offset
    features_w_offset = dataset.features_w_offset(index,:,:,:);

    % features_wo_offset
    features_wo_offset = dataset.features_wo_offset(index,:,:,:);

    % labels_gaussian_2d
    labels_gaussian_2d = dataset.labels_gaussian_2d(index,:,:);

    % labels
    labels = dataset.labels(index,:);
    x_values = dataset.x_values;
    y_values = dataset.y_values;

    save(fullfile(save_dir, file_name),...
        'features_w_offset',...
        'features_wo_offset',...
        'labels_gaussian_2d',...
        'labels',...
        'index',...
        'x_values',...
        'y_values',...
        '-v7.3');
    fprintf('data saved in %s\n', fullfile(save_dir, file_name))
end