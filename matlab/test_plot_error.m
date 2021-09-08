%% load reuslts data
dloc_result_path = "/Users/Charlie/Documents/Work_School/UCSD/RESEARCH/temp_data/phone_4AP/analysis/DLoc_results/decoder_test_result_epoch_best.mat";
dloc_result = load(dloc_result_path);

%% convert cell to matrix
dloc_error = cell2mat(dloc_result.error);

%% error cdf 
figure;
h_dloc = cdfplot(dloc_error); hold on;
h_dloc.LineWidth = 2;
h_fft = cdfplot(act_fin_error); 
h_fft.LineWidth = 2;
legend('DLoc','svd+fft+tracking','FontSize',12,'Location','best');
hold off;
title("DLoc vs svd+fft Localization Error",'FontSize',12);
xlim([0 8])