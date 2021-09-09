function labels_gaussian = get_gaussian_labels(labels, sigma, x_vals, y_vals)
    % Generate ground truth label for training. Produce prob value for each
    % grid points using a 2d gaussian distribution. 
    % labels: Nx2, xy ground truth location
    % Args:
    % x_vals: 1 X N_x arrays of x grid values
    % y_vals: 1 X N_y arrays of y grid values
    % Return:
    % labels_gaussian: N_points x N_y x N_x, XY images output of 2d gaussian pdf.
    
    n_xlabels = length(x_vals);
    n_ylabels =length(y_vals);
    labels_gaussian = zeros(size(labels,1),n_ylabels,n_xlabels);
    map_X = repmat(x_vals,n_ylabels,1);
    map_Y = repmat(y_vals',1,n_xlabels);
    n_points = size(labels,1);
    
    for i=1:n_points
        d = (map_X-labels(i,1)).^2+(map_Y-labels(i,2)).^2;
        cur_gaussian = exp(-d/sigma/sigma);%*1/sqrt(2*pi)/output_sigma;        
        labels_gaussian(i,:)=cur_gaussian(:);
    
    end
end



