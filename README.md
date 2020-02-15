# DLoc MATLAB data processing

# DLoc PyTorch Implementation
Deep Learning architecture codes for Indoor WiFi based localization

The master\_enc\_2dec.py function calls the main\_enc\_2dec.py function with the appropriate parameter files from params.py
The master\_enc\_dec.py function calls the main\_enc\_dec.py function with the appropriate parameter files from params.py

#### Note:
+ The data path files assume the data in the folder ../datasets/
+ The parameters to tune are included in the params.py for each iteration.
+ Examples parameter files for DLoc's (e19) architecture and DLoc without Consistency Decoder architecture (e15)

# Dataset Release
## Different Setups
- **July16**: 8m X 5m setup with 3 APs in Atkinson hall ground floor for data collected on July 16, 2019.
- **July18**: 8m X 5m setup with 3 APs in Atkinson hall ground floor for data collected on July 18, 2019.
- **July22_2_ref**: 8m X 5m setup with 3 APs and 2 additonal reflectors (*a huge aluminium plated board*) placed in Atkinson hall ground floor for data collected on July 2, 2019.
- **jacobs_Jul28**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on July 28, 2019.
- **jacobs_Jul28_2**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on July 28, 2019, one hour after **jacobs_Jul28**
- **jacobs_Aug16_1**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on August 16, 2019 with extra furniture placed randomly
- **jacobs_Aug16_3**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on August 16, 2019 with extra furniture placed randomly
- **jacobs_Aug16_4_ref**: 18m X 8m setup with 4 APs in Jacobs hall ground floor for data collected on August 16, 2019 with extra furniture placed randomly with an added reflector (*a huge aluminium plated board*)

---

We provide both the CSI data for all the above setups and the post-prcessed features for running our DLoc network

## Channels

The CSI data is placed under the **channels** folder and are named as *channels_<setup_name_from_above>.mat*. These MATLAB files are stored using *HDF5* file structure and contain the following variables:

- **channels**: *[ n_datapoints x n_frequency x n_ant X n_ap ]* 4D complex channel matrix.
- **RSSI**: *[ n_datapoints x n_ap ]* 2D recieved signal strenght matrix.
- **labels**: *[ n_datapoints x 2 ]* 2D XY labels.
- **opt**: various options specific for the data generated
	-*opt.freq* : *[n_frequencyx1]* 1D vector that describes the frequency of the subcarriers
	-*opt.lambda*: *[n_frequencyx1]* 1D vector that describes the wavelength of the subcarriers
	-*ant_sep*: antenna separation used on all of our APs
- **ap**: *n_ap* cell matrix. Each element corresposning to *[ n_ant x 2 ]* XY locations of the n_ant on each AP.
- **ap_aoa**: *[ n_ap x 1]* vectors that contains the rotation that needs to be added to the AoA measured at each AP (assumes that the AoA is measured about the normal to the AP's antenna array)
- **d1**: The sampled x-axis of the space under consideration
- **d2**: The sampled y-axis of the space under consideration

## Datasets

The CSI data is placed under the **datasets** folder and are named as *features_<setup_name_from_above>.mat*. These MATLAB files are stored using *HDF5* file structure and contain the following variables:

- **features_with_offset**: *[ n_datapoints x n_ap x n_d1_points X n_d2_points ]* 4D feature matrix for n_ap **with offsets** in time
- **features_without_offset**: *[ n_datapoints x n_ap x n_d1_points X n_d2_points ]* 4D feature matrix for n_ap **without offsets** in time
- **labels_gaussian_2d**: *[ n_datapoints x n_d1_points X n_d2_points ]* 3D labels matrix that contisn the target images for the location network.
- **labels**: *[ n_datapoints x 2 ]* 2D XY labels.
