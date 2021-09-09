function xy_labels = convert_img_to_xy(gaussian_labels,x_values,y_values)
    % Convert XY images to XY coordinate values.
    % gaussian_labels: [n_points, y_size, x_size], gaussian XY images
    % Args:
    % x_values: [1, x_size], x axis grid values
    % y_values: [1, y_size], y axis grid values
    % Return:
    % xy_labels: [n_points, 2], XY coordinate values. Mean of gaussian.
    
    n_points = size(gaussian_labels,1);
    xy_labels = zeros(n_points,2);
    
    for i=1:n_points
        img = squeeze(gaussian_labels(i,:,:));
        [~,linear_idx] = max(img(:));
        [row_idx, col_idx] = ind2sub(size(img), linear_idx);
        xy_labels(i,1) = x_values(col_idx);
        xy_labels(i,2) = y_values(row_idx);
    end
end