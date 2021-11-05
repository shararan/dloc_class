function DP=compute_multipath_profile2d_fast_edit(h,theta_vals,d_vals,opt)
    % Our Vectorized implementation 2D FFT from DLoc/BLoc
    % INPUT
    % h: 4 times 234 matrix of csi measurements (with slope across subcarriers
    % removed)
    % theta_vals: values of time where the profile has to be evaluate
    % d_vals: values of distance where the profile has to be evaluated
    % opt: optional values including the frequency subarriers, wavelength
    % OUTPUT
    % p: is a length(theta_vals) times length(d_vals) array of complex
    % values
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Steps
    % Create the Signal matrix
    % Find eigen values
    % Detect Noise Subspace
    % Compute projects on the noise subspace
    freq_cent = median(opt.freq);
    const = 1j*2*pi/(3e8);
    const2 = 1j*2*pi*opt.ant_sep*freq_cent/(3e8);
    h = h.';
    d_rep = const*(opt.freq'.*repmat(d_vals,length(opt.freq),1));
    temp = h*exp(d_rep);
    theta_rep = const2*((1:size(h,1)).*repmat(sin(theta_vals'),1,size(h,1)));
    DP = exp(theta_rep)*(temp);
end