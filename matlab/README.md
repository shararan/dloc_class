# Prepare CSI Data for Training

## CSI -> training & testing data

To generate training and testing data for DLoc from raw CSI data, run matlab script:

```setup
generate_features_labels_from_channels.m
```  
### Script Input
Make sure to change the following lines for each dataset:
```
data_path = "path/to/csi_data.mat";
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
### Script Output
The script will generate two mat files: `data_train.mat` and `data_test.mat`. Each `mat` file will contain the following attributes:
```
'features_w_offset',...     % input to the network
'features_wo_offset',...    % ground truth label for consistency decoder       
'labels_gaussian_2d',...    % ground truth label for location decoder
'labels',...                % xy location ground truth
'index',...                 % index corresponds to training/testing data
'x_values',...              % x axis grid value
'y_values',...              % y axis grid value
```
