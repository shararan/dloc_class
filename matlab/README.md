# Prepare CSI Data for Training

 

## CSI -> training & testing data

To generate training and testing data for DLoc from raw CSI channels, run matlab script:

```setup
generate_features_labels_from_channels.m
```  

Make sure to change the following lines for each dataset:
```
data_path = "path/to/csi_data.mat";
x_range_max = integer;   % size of the experiment arena alone x direction
y_range_max = integer;   % size of the experiment arena alone y direction
```

**Note**: `csi_data.mat` must contain the following fields:
```
'channels', ...       % size = [n_point,n_sub,n_ant,n_ap], compensated csi data 
'labels', ...         % size = [n_points, 2], xy ground truth labels
'real_tof', ...       % size = [n_points, n_ap], real time of flight in m
'theta_vals',...      % aoa search space in radians
'd_vals',...          % tof search space in m
'opt',...             % struct, contain constants like freq, bandwidth, etc
'ap',...              % cell, xy coordinates of antennas of all access point
'dataset_name',...    % str, name of dataset
'd_pred');            % tof prediction in meter
```

