function [train_idxs, test_idxs] = split_train_test(labels, bboxs)
    %{
    Partition training and testing data. Test data is selected based on
    provided bounding box.
    
    Args:
        labels (matrix): [n_samples, 2], xy coordinates of all samples.
        bboxs (cell): {[x_min, x_max, y_min, y_max],...} bounding box for selecting test data.
    
    Return:
        train_idxs (vector): [n_train_sample,1], index for training data
        test_idxs (vector): [n_test_sample,1], index for test data
    %}
    test_mask = zeros(size(labels,1), 1);
    
    for i = 1:length(bboxs)
        bbox = bboxs{i};
        idx_box = (labels(:,1) >=  bbox(1)) & (labels(:,1) <=  bbox(2)) & ...
                  (labels(:,2) >=  bbox(3)) & (labels(:,2) <=  bbox(4));
        test_mask = test_mask | idx_box;
    end
    
    if sum(test_mask) == 0
        Error("No sample inside bounding box provided");
    end
    
    idx_all = 1:size(labels,1);
    test_idxs = idx_all(test_mask);
    train_idxs = idx_all(~test_mask);
    
    fprintf('Test sample: %f of total data.\n', length(test_idxs)/length(idx_all));
end