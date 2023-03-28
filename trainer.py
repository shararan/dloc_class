'''
Scripts for the training and testing functions
train() function is called for training the network
test() function is called to evaluate the network
Both the function logs and saves the results in the files 
as mentioned in the params.py file
'''
from curses import raw
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

def train(model, train_loader, test_loader, experiment):
    """
    Training pipeline

    Args:
        model (torch.module): pytorch model
        train_loader (torch.dataloader): dataloader
        test_loader (torch.dataloader): dataloader
    """
    # set data index
    offset_output_index= "features_wo_offset"
    input_index= "features_w_offset"
    output_index= "labels_cl"
    
    # initialization
    total_steps = 0
    print('Training called')
    stopped_count = 0
    curr_best_error = 100
    # curr_best_error = 0
    stopping_count = 0
    early_stop_epoch = 5
    early_stop_epoch_ct = 2
    labels = []
    xy_labels = []
    trainIndicies = []
    overall_loss_count = 0

    for epoch in range(model.opt.starting_epoch_count+1, model.opt.n_epochs+1): # opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_offset_loss = 0
        error = []

        for i, (data, idx) in enumerate(train_loader):
            
            total_steps += model.opt.batch_size
            data['features_w_offset'] = torch.squeeze(data['features_w_offset'], dim=0)
            data['labels_gaussian_2d'] = torch.squeeze(data['labels_gaussian_2d'], dim=0)
            data['features_wo_offset'] = torch.squeeze(data['features_wo_offset'], dim=0)
            # print(f'Labels: {data["features_w_offset"]}')
            # print(f'Input Data Size: {data[input_index].shape}')
            # print(f'Target Size: {data[output_index].shape}')
            if opt_exp.n_decoders == 2:
                model.set_input(data[input_index], data[output_index], data[offset_output_index], shuffle_channel=False)
            elif opt_exp.n_decoders == 1:
                model.set_input(data[input_index], data[output_index].squeeze(1), shuffle_channel=False)
            elif opt_exp.n_decoders == 0:
                model.set_data(data[input_index], data[output_index], shuffle_channel=False)
            model.optimize_parameters()

            if opt_exp.n_decoders == 2 or opt_exp.n_decoders == 1:
                dec_outputs = model.decoder.output
            elif opt_exp.n_decoders == 0:
                dec_outputs = model.output
            # print(f"dec_outputs size is : {dec_outputs.shape}")
            error.extend(localization_error(dec_outputs.data.cpu().numpy(), data['labels_cl'].cpu().numpy(), scale=0.1, ap_aoas=data['ap_aoas'].cpu().numpy(), ap_locs=data['ap_locs'].cpu().numpy()))

            if epoch == 1:
                labels.extend(data[output_index].cpu().numpy())
                xy_labels.extend(data['labels_cl'].cpu().numpy())
                trainIndicies.extend(idx.cpu().numpy())

            if opt_exp.n_decoders == 0:
                write_log([str(model.loss.item())], model.model_name, log_dir=model.opt.log_dir, log_type='loss')
                overall_loss_count += 1
                experiment.log_metric("overall_loss", model.loss.item(), overall_loss_count)
            if opt_exp.n_decoders == 1:
                write_log([str(model.decoder.loss.item())], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='loss')
            if opt_exp.n_decoders == 2:
                write_log([str(model.offset_decoder.loss.item())], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='offset_loss')
            
            if opt_exp.n_decoders >= 1:
                if total_steps % model.decoder.opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                    model.save_networks('latest')
            elif opt_exp.n_decoders == 0:
                if total_steps % model.opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                    model.save_networks('latest')

            if opt_exp.n_decoders == 2 or opt_exp.n_decoders == 1:
                epoch_loss += model.decoder.loss.item()
            elif opt_exp.n_decoders == 0:
                epoch_loss += model.loss.item()

            if opt_exp.n_decoders == 2:
                epoch_offset_loss += model.offset_decoder.loss.item()
            if i%20==0:
                print(f"Batch: {i}, Time Taken: {(time.time() - epoch_start_time):.2f} seconds")

        print("Time taken for the epoch %d is: %.5s secs" % (epoch,  time.time() - epoch_start_time))

        median_error_tr = np.median(error)
        error_90th_tr = np.percentile(error,90)
        error_99th_tr = np.percentile(error,99)
        epoch_loss /= i

        # train_accuracy = np.mean(error)
        print(f"train_med_error for epoch {epoch} is {median_error_tr}")

        if opt_exp.n_decoders == 2:
            epoch_offset_loss /= i
        if opt_exp.n_decoders == 1:
            write_log([str(epoch_loss)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='epoch_decoder_loss')
            experiment.log_metric("epoch_decoder_loss",epoch_loss, epoch)
        if opt_exp.n_decoders == 0:
            write_log([str(epoch_loss)], model.model_name, log_dir=model.opt.log_dir, log_type='epoch_decoder_loss')
            experiment.log_metric("epoch_decoder_loss",epoch_loss, epoch)
        if opt_exp.n_decoders == 2:
            write_log([str(epoch_offset_loss)], model.offset_decoder.model_name, log_dir=model.offset_decoder.opt.log_dir, log_type='epoch_offset_decoder_loss')
            experiment.log_metric("epoch_offset_decoder_loss",epoch_offset_loss, epoch)

        if opt_exp.n_decoders == 1 or opt_exp.n_decoders == 2:
            write_log([str(median_error_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_med_error')
            experiment.log_metric("train_med_error", median_error_tr, epoch)
            write_log([str(error_90th_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_90th_error')
            experiment.log_metric("train_90th_error",error_90th_tr, epoch)
            write_log([str(error_99th_tr)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='train_99th_error')
            experiment.log_metric("train_99th_error",error_99th_tr, epoch)

        elif opt_exp.n_decoders ==0:
            write_log([str(median_error_tr)], model.model_name, log_dir=model.opt.log_dir, log_type='train_median_error')
            experiment.log_metric("train_median_error",median_error_tr, epoch)
            write_log([str(error_90th_tr)], model.model_name, log_dir=model.opt.log_dir, log_type='train_90th_error')
            experiment.log_metric("train_90th_error",error_90th_tr, epoch)
            write_log([str(error_99th_tr)], model.model_name, log_dir=model.opt.log_dir, log_type='train_99th_error')
            experiment.log_metric("train_99th_error",error_99th_tr, epoch)

        if (epoch==1):
            _, median_error = valid(model, epoch, test_loader, experiment, save_output=True)
            # _, model_accuracy = valid(model, epoch, test_loader, experiment, save_output=True)
        else:
            _, new_med_error = valid(model, epoch, test_loader, experiment, save_output=True)
            if (median_error >= new_med_error):
                stopping_count = stopping_count+1
                median_error = new_med_error
            # if (model_accuracy >= new_model_accuracy):
            #     stopping_count = stopping_count+1
            #     model_accuracy = new_model_accuracy

        if opt_exp.n_decoders == 2 or opt_exp.n_decoders == 1:

            if epoch % model.encoder.opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)
            if (stopping_count==early_stop_epoch):
                stopped_count += 1
                stopping_count = 0
                print('Saving best model at %d epoch' %(epoch-early_stop_epoch))
                model.load_networks(epoch-early_stop_epoch)
                model.save_networks('best')
                model.load_networks('latest')
                if(stopped_count==early_stop_epoch_ct and curr_best_error >= median_error):
                    break
                elif(stopped_count==early_stop_epoch_ct):
                    stopped_count -= 1
                write_log([str(epoch)], model.decoder.model_name, log_dir=model.decoder.opt.log_dir, log_type='best_epoch')
                experiment.log_metric("best_epoch", epoch)
            
        elif opt_exp.n_decoders == 0:

            if epoch % model.opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)
            if (stopping_count==early_stop_epoch):
                stopped_count += 1
                stopping_count = 0
                print('Saving best model at %d epoch' %(epoch-early_stop_epoch))
                model.load_networks(epoch-early_stop_epoch)
                model.save_networks('best')
                model.load_networks('latest')
                if(stopped_count==early_stop_epoch_ct and curr_best_error>=median_error):
                    break
                elif(stopped_count==early_stop_epoch_ct):
                    stopped_count -= 1

                write_log([str(epoch)], model.model_name, log_dir=model.opt.log_dir, log_type='best_epoch')
                experiment.log_metric("best_epoch", epoch)



        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, model.opt.n_epochs, time.time() - epoch_start_time))
        if opt_exp.n_decoders == 0:
            model.update_learning_rate()
        if opt_exp.n_decoders == 1:
            model.decoder.update_learning_rate()
            model.encoder.update_learning_rate()
        if opt_exp.n_decoders == 2:
            model.offset_decoder.update_learning_rate()

    if opt_exp.n_decoders == 1 or opt_exp.n_decoders == 2:
        save_dir = model.decoder.results_save_dir # default save directory
    elif opt_exp.n_decoders == 0:
        save_dir = model.results_save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    save_path = f"{save_dir}/train_labels.h5"
    h5_options = hdf5storage.Options(store_python_metadata=True, matlab_compatible=True)
    hdf5storage.writes(
        mdict={"labels" : labels, "xy_labels" : xy_labels, "indicies" : trainIndicies}, 
        filename=save_path,
        options=h5_options
    )
    print(f"Train Labels saved in {save_path}")


def valid(model, epoch, test_loader, experiment, save_output=True, save_name="decoder_valid_result", save_dir="", log=True):
    """Test and evaluation pipeline

    Args:
        model (torch.module): pytorch model
        test_loader (torch.dataloader): dataloader
        save_output (bool, optional): whether to save output to mat file. Defaults to True.
        save_name (str, optional): name of the mat file. Defaults to "decoder_test_result".
        save_dir (str, optional): directory where output mat file is saved. Defaults to "".
        log (bool, optional): whether to log output. Defaults to True.

    Returns:
        tuple: (total_loss -> float, median_error -> float)
    """
    print('Evaluation Called')
    model.eval()

    # set data index
    offset_output_index= "features_wo_offset"
    input_index= "features_w_offset"
    output_index= "labels_cl"

    # create containers
    generated_outputs = []
    offset_outputs = []
    total_loss = 0
    total_offset_loss = 0
    error =[]
    validIndicies = []

    for i, (data, idx) in enumerate(test_loader):
        validIndicies.extend(idx.cpu().numpy())
        data['features_w_offset'] = torch.squeeze(data['features_w_offset'], dim=0)
        data['labels_gaussian_2d'] = torch.squeeze(data['labels_gaussian_2d'], dim=0)
        data['features_wo_offset'] = torch.squeeze(data['features_wo_offset'], dim=0)
        if opt_exp.n_decoders == 2:
                model.set_input(data[input_index], data[output_index], data[offset_output_index], shuffle_channel=False)
        elif opt_exp.n_decoders == 1:
                model.set_input(data[input_index], data[output_index].squeeze(1), shuffle_channel=False)
        elif opt_exp.n_decoders == 0:
            model.set_data(data[input_index], data[output_index], shuffle_channel=False)

        model.test()

        # get model outputs
        if opt_exp.n_decoders == 1:
            gen_outputs = model.decoder.output  # gen_outputs.size = (N,1,H,W)
        if opt_exp.n_decoders == 2:
            gen_outputs = model.decoder.output  # gen_outputs.size = (N,1,H,W)
            off_outputs = model.offset_decoder.output # off_outputs.size = (N,n_ap,H,W)

        if opt_exp.n_decoders == 0:
            gen_outputs = model.output

        generated_outputs.extend(gen_outputs.data.cpu().numpy())
        if opt_exp.n_decoders == 2:
            offset_outputs.extend(off_outputs.data.cpu().numpy())
        error.extend(localization_error(gen_outputs.data.cpu().numpy(),data['labels_cl'].cpu().numpy(),scale=0.1, ap_aoas=data['ap_aoas'].cpu().numpy(), ap_locs=data['ap_locs'].cpu().numpy()))
        if opt_exp.n_decoders == 2 or opt_exp.n_decoders == 1:
            total_loss += model.decoder.loss.item()
        elif opt_exp.n_decoders == 0:
            total_loss += model.loss.item()
        if opt_exp.n_decoders == 2:
            total_offset_loss += model.offset_decoder.loss.item()

    total_loss /= i
    if opt_exp.n_decoders == 2:
        total_offset_loss /= i
    # valid_accuracy = np.mean(error)

    median_error = np.median(error)
    nighty_percentile_error = np.percentile(error,90)
    error_99th = np.percentile(error,99)

    if log and (opt_exp.n_decoders == 1 or opt_exp.n_decoders == 2):
        write_log([str(median_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='valid_median_error')
        experiment.log_metric("valid_median_error", median_error, epoch)
        write_log([str(nighty_percentile_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_90_error')
        experiment.log_metric("valid_90_error",nighty_percentile_error, epoch)
        write_log([str(error_99th)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_99_error')
        experiment.log_metric("valid_99_error",error_99th, epoch)
        write_log([str(total_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
        experiment.log_metric("valid_loss",total_loss, epoch)
        if opt_exp.n_decoders == 2:
            write_log([str(total_offset_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_offset_loss')
            experiment.log_metric("valid_offset_loss",total_offset_loss, epoch)

    elif log and opt_exp.n_decoders == 0:
        write_log([str(median_error)], model.model_name, log_dir=model.opt.log_dir, log_type='test_median_error')
        experiment.log_metric("valid_median_error",median_error, epoch)
        write_log([str(nighty_percentile_error)], model.model_name, log_dir=model.opt.log_dir, log_type='test_90_error')
        experiment.log_metric("valid_90_error",nighty_percentile_error, epoch)
        write_log([str(error_99th)], model.model_name, log_dir=model.opt.log_dir, log_type='test_99_error')
        experiment.log_metric("valid_99_error",error_99th, epoch)
        write_log([str(total_loss)], model.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
        experiment.log_metric("valid_loss",total_loss, epoch)

    if save_output and epoch%5==0:
        if not save_dir and opt_exp.n_decoders >= 1:
            save_dir = model.decoder.results_save_dir # default save directory
        elif not save_dir and opt_exp.n_decoders == 0:
            save_dir = model.results_save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        save_path = f"{save_dir}/{save_name}.h5"
        h5_options = hdf5storage.Options(store_python_metadata=True, matlab_compatible=True)
        hdf5storage.writes(mdict={"outputs":generated_outputs,"wo_outputs":offset_outputs, "error": error, "indicies" : validIndicies}, 
                          filename=save_path,
                          options=h5_options)
                          # appendmat=True, 
                          # format='7.3',truncate_existing=True)
        print(f"Result saved in {save_path}")
    return total_loss, median_error


def test(model, test_loader, experiment, save_output=True, save_name="decoder_test_result", save_dir="", log=True):
    """Test and evaluation pipeline

    Args:
        model (torch.module): pytorch model
        test_loader (torch.dataloader): dataloader
        save_output (bool, optional): whether to save output to mat file. Defaults to True.
        save_name (str, optional): name of the mat file. Defaults to "decoder_test_result".
        save_dir (str, optional): directory where output mat file is saved. Defaults to "".
        log (bool, optional): whether to log output. Defaults to True.

    Returns:
        tuple: (total_loss -> float, median_error -> float)
    """
    print('Evaluation Called')
    # model.eval()

    # set data index
    offset_output_index= "features_wo_offset"
    input_index= "features_w_offset"
    output_index= "labels_cl"

    # create containers
    generated_outputs = []
    output_labels = []
    features_w_offset = []
    features_wo_offset = []
    labels = []
    xy_labels = []
    offset_outputs = []
    total_loss = 0
    total_offset_loss = 0
    error =[]
    testIndicies = []

    for i, (data, idx) in enumerate(test_loader):
        testIndicies.extend(idx.cpu().numpy())
        data['features_w_offset'] = torch.squeeze(data['features_w_offset'], dim=0)
        data['labels_gaussian_2d'] = torch.squeeze(data['labels_gaussian_2d'], dim=0)
        data['features_wo_offset'] = torch.squeeze(data['features_wo_offset'], dim=0)
        if opt_exp.n_decoders == 2:
                model.set_input(data[input_index], data[output_index], data[offset_output_index], shuffle_channel=False)
        elif opt_exp.n_decoders == 1:
                model.set_input(data[input_index], data[output_index].squeeze(1), shuffle_channel=False)
        elif opt_exp.n_decoders == 0:
                model.set_data(data[input_index], data[output_index], shuffle_channel=False)

        model.test()

        # get model outputs
        if opt_exp.n_decoders == 1:
            gen_outputs = model.decoder.output  # gen_outputs.size = (N,1,H,W)
        if opt_exp.n_decoders == 2:
            gen_outputs = model.decoder.output  # gen_outputs.size = (N,1,H,W)
            off_outputs = model.offset_decoder.output # off_outputs.size = (N,n_ap,H,W)
        if opt_exp.n_decoders == 0:
            gen_outputs = model.output

        generated_outputs.extend(gen_outputs.data.cpu().numpy())
        features_w_offset.extend(data[input_index].cpu().numpy())
        features_wo_offset.extend(data[offset_output_index].cpu().numpy())
        labels.extend(data[output_index].cpu().numpy())
        xy_labels.extend(data['labels_xy'].cpu().numpy())
        if opt_exp.n_decoders == 2:
            offset_outputs.extend(off_outputs.data.cpu().numpy())
        error.extend(localization_error(gen_outputs.data.cpu().numpy(),data['labels_cl'].cpu().numpy(),scale=0.1, ap_aoas=data['ap_aoas'].cpu().numpy(), ap_locs=data['ap_locs'].cpu().numpy()))
        if opt_exp.n_decoders >= 1:
            total_loss += model.decoder.loss.item()
        elif opt_exp.n_decoders == 0:
            total_loss += model.loss.item()

        if opt_exp.n_decoders == 2:
            total_offset_loss += model.offset_decoder.loss.item()
    
    for i in range(len(generated_outputs)):
        output_labels.append(np.argmax(generated_outputs[i]))

    total_loss /= i
    if opt_exp.n_decoders == 2:
        total_offset_loss /= i
    # test_accuracy = np.mean(error)
    median_error = np.median(error)
    nighty_percentile_error = np.percentile(error,90)
    error_99th = np.percentile(error,99)

    if log and opt_exp.n_decoders >= 1:
        # write_log([str(test_accuracy)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_classification_accuracy')
        # experiment.log_metric("test_classification_accuracy", test_accuracy)
        write_log([str(median_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_median_error')
        experiment.log_metric("test_median_error",median_error)
        write_log([str(nighty_percentile_error)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_90_error')
        experiment.log_metric("test_90_error",nighty_percentile_error)
        write_log([str(error_99th)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_99_error')
        experiment.log_metric("test_99_error",error_99th)
        write_log([str(total_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
        experiment.log_metric("test_loss",total_loss)
        if opt_exp.n_decoders == 2:
            write_log([str(total_offset_loss)], model.decoder.model_name, log_dir=model.opt.log_dir, log_type='test_offset_loss')
            experiment.log_metric("test_offset_loss",total_offset_loss)

    elif log and opt_exp.n_decoders == 0:
        write_log([str(median_error)], model.model_name, log_dir=model.opt.log_dir, log_type='test_median_error')   
        experiment.log_metric("test_median_error",median_error)
        write_log([str(nighty_percentile_error)], model.model_name, log_dir=model.opt.log_dir, log_type='test_90_error')
        experiment.log_metric("test_90_error",nighty_percentile_error)
        write_log([str(error_99th)], model.model_name, log_dir=model.opt.log_dir, log_type='test_99_error')
        experiment.log_metric("test_99_error",error_99th)
        write_log([str(total_loss)], model.model_name, log_dir=model.opt.log_dir, log_type='test_loss')
        experiment.log_metric("test_loss",total_loss)

    if save_output:
        if not save_dir and opt_exp.n_decoders >= 1:
            save_dir = model.decoder.results_save_dir # default save directory
        elif not save_dir and opt_exp.n_decoders == 0:
            save_dir = model.results_save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        save_path = f"{save_dir}/{save_name}.mat"
        # h5_options = hdf5storage.Options(store_python_metadata=True, matlab_compatible=True)
        hdf5storage.savemat(save_path,
                            mdict={"outputs":output_labels,
                                    "wo_outputs":offset_outputs,
                                    "labels": labels, 
                                    "error": error, 
                                    "indicies" : testIndicies},
                          appendmat=True, 
                          format='7.3',truncate_existing=True)
        print(f"result saved in {save_path}")
    return total_loss, median_error, error #, nighty_percentile_error, error_99th