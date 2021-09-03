import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
import numpy as np  
import os
import time
import hdf5storage
from utils import *
from Generators import *
from params import *

def train(model, train_loader, test_loader, input_index=1, output_index=2, offset_output_index=0):
    total_steps = 0
    print('Training called')
    # generator_outputs = []
    # location_outputs = []
    stopping_count = 0
    for epoch in range(model.opt.starting_epoch_count+1, model.opt.n_epochs+1): # opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_offset_loss = 0
        error =[]

        for i, data in enumerate(train_loader):
            total_steps += model.opt.batch_size
            model.set_input(data[input_index], data[output_index], data[offset_output_index], shuffle_channel=False)
            model.optimize_parameters()
            dec_outputs = model.decoder.output
            error.extend(localization_error(dec_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))

            write_log([str(model.decoder.loss.item())], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='loss')
            write_log([str(model.offset_decoder.loss.item())], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='offset_loss')
            if total_steps % model.decoder.opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            epoch_loss += model.decoder.loss.item()
            epoch_offset_loss += model.offset_decoder.loss.item()

        median_error_tr = np.median(error)
        error_90th_tr = np.percentile(error,90)
        error_99th_tr = np.percentile(error,99)
        nighty_percentile_error_tr = np.percentile(error,90)
        epoch_loss /= i
        epoch_offset_loss /= i
        write_log([str(epoch_loss)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='epoch_decoder_loss')
        write_log([str(epoch_offset_loss)], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='epoch_offset_decoder_loss')
        write_log([str(median_error_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_median_error')
        write_log([str(error_90th_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_90th_error')
        write_log([str(error_99th_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_99th_error')
        write_log([str(nighty_percentile_error_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_90_error')
        if (epoch==1):
            min_eval_loss, median_error = test(model, test_loader, input_index=input_index, output_index=output_index, offset_output_index=offset_output_index, save_output=True)
        else:
            new_eval_loss, new_med_error = test(model, test_loader, input_index=input_index, output_index=output_index, offset_output_index=offset_output_index, save_output=True)
            if (median_error>=new_med_error):
                stopping_count = stopping_count+1
                median_error = new_med_error

        # generated_outputs = temp_generator_outputs
        if epoch % model.encoder.opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            if (stopping_count==2):
                print('Saving best model at %d epoch' %(epoch))
                model.save_networks('best')
                stopping_count=0

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, model.decoder.opt.niter + model.decoder.opt.niter_decay, time.time() - epoch_start_time))
        model.decoder.update_learning_rate()
        model.encoder.update_learning_rate()
        model.offset_decoder.update_learning_rate()


def test(model, test_loader, input_index=1, output_index=2,  offset_output_index=0, save_name="decoder_test_result", save_output=True, log=True):
    print('Evaluation Called')
    model.eval()
    generated_outputs = []
    offset_outputs = []
    total_loss = 0
    total_offset_loss = 0
    error =[]
    for i, data in enumerate(test_loader):
            model.set_input(data[input_index], data[output_index], data[offset_output_index],convert_enc=True, shuffle_channel=False)
            model.test()

            # get model outputs
            gen_outputs = model.decoder.output  # gen_outputs.size = (N,1,H,W)
            off_outputs = model.offset_decoder.output # off_outputs.size = (N,n_ap,H,W)

            generated_outputs.extend(gen_outputs.data.cpu().numpy())
            offset_outputs.extend(off_outputs.data.cpu().numpy())
            error.extend(localization_error(gen_outputs.data.cpu().numpy(),data[output_index].cpu().numpy(),scale=0.1))
            total_loss += model.decoder.loss.item()
            total_offset_loss += model.offset_decoder.loss.item()
    total_loss /= i
    total_offset_loss /= i
    median_error = np.median(error)
    nighty_percentile_error = np.percentile(error,90)
    error_99th = np.percentile(error,99)

    if log:
        write_log([str(median_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_median_error')
        write_log([str(nighty_percentile_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_90_error')
        write_log([str(error_99th)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_99_error')
        write_log([str(total_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
        write_log([str(total_offset_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_offset_loss')

    if save_output:
        if not os.path.exists(model.decoder.results_save_dir):
            os.makedirs(model.decoder.results_save_dir, exist_ok=True)
        save_path = f"{model.decoder.results_save_dir}/{save_name}.mat"
        hdf5storage.savemat(save_path,
            mdict={"outputs":generated_outputs,"wo_outputs":offset_outputs, "error": error}, 
            appendmat=True, 
            format='7.3',
            truncate_existing=True)
        print(f"result saved in {save_path}")
    return total_loss, median_error