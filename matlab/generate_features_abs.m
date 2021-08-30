function features = generate_features_abs(channels,ap,theta_vals,d_vals,d1,d2,opt)
    %{
    Convert CSI channels to image for training.
    Args:
        channels (matrix): size = [n_pkt, n_sub, n_rx, n_ap], csi channel data
        ap (cell): xy coordinates of AP antennas. Each cell is n_rx x 2.
        theta_vals (vector): AoA search space in radians. 
        d_vals (vector): ToF search space in m.
        d1 (vector): x axis of image.
        d2 (vector): y axis of image.
        opt (struct): stuct that contains constants like freq, bandwidth, etc.
    
    Returns:
        features (matrix): size = [n_pkt, n_ap, length(d2), length(d1)].
        Profile cartesian coordinates. 
    %}
    
    n_ap=length(ap);
    n_ant=size(ap{1},1);
    
    % channels_rel = zeros(n_lambda,n_ap,n_ant,n_ap-1);
    features = zeros(n_ap,length(d2),length(d1));
    % feature_idx=1;
    
    for j=1:n_ap
        P = compute_multipath_profile2d_fast_edit(squeeze(channels(:,:,j)),theta_vals,d_vals,opt);
        P_out = convert_spotfi_to_2d(P,theta_vals,d_vals,d1,d2,ap{j}); % ap{j}: 4x2, antenna xy coordinates
        features(j,:,:) = abs(P_out);
    end
end