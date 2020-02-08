
import os
from easydict import EasyDict as edict
import json



opt_exp = edict()

opt_exp.isTrain = False
opt_exp.continue_train = False #action='store_true', help='continue training: load the latest model')
opt_exp.starting_epoch_count = 'best' #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_exp.isTrainGen = False
opt_exp.isTrainLoc = True
opt_exp.isFrozen = False
opt_exp.isFrozen_gen = False
opt_exp.n_epochs = 50
opt_exp.data = "rw_to_rw"
opt_exp.gpu_ids = ['0','1','2']

opt_exp.phase = 'rw_test' #type=str, default='train', help='train, val, test, etc')
opt_exp.name = 'e15_realgen_train' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

opt_exp.checkpoints_dir = './checkpoints' + "/" + opt_exp.name#type=str, default='./checkpoints', help='models are saved here')
opt_exp.results_dir = './results' + "/" + opt_exp.name
opt_exp.log_dir = "./logs" + "/" + opt_exp.name
opt_exp.batch_size = 32


# In[13]:
# define base_options
opt_gen = edict()

# hyperparams
# dataroot #'path to images (should have subfolders trainA, trainB, valA, valB, etc)')
opt_gen.parent_exp = opt_exp
opt_gen.batch_size = 32 #type=int, default=1, help='input batch size')
opt_gen.ngf = 64 #type=int, default=64, help='# of gen filters in first conv layer')
opt_gen.base_model = 'resnet2_nblocks' #type=str, default='resnet_9blocks', help='selects model to use for netG')
opt_gen.resnet_blocks = 9
opt_gen.net = 'G' #type=str, default='resnet_9blocks', help='selects model to use for netG')
opt_gen.no_dropout = False #action='store_true', help='no dropout for the generator')
opt_gen.init_type = 'xavier' #type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
opt_gen.init_gain = 0.02 #type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
opt_gen.norm = 'instance' #type=str, default='instance', help='instance normalization or batch normalization')
opt_gen.beta1 = 0.5 #type=float, default=0.5, help='momentum term of adam')
opt_gen.lr = 0.00001 #type=float, default=0.0002, help='initial learning rate for adam')
opt_gen.lr_policy = 'lambda' #type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
opt_gen.lr_decay_iters = 50 #type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
opt_gen.lambda_L = 1 # weightage given to the Generator
opt_gen.lambda_cross = 1e-5
opt_gen.lambda_reg = 1e-3
opt_gen.weight_decay = 1e-4

opt_gen.input_nc = 4 #type=int, default=3, help='# of input image channels')
opt_gen.output_nc = 1 #type=int, default=3, help='# of output image channels')
opt_gen.save_latest_freq = 5000 #type=int, default=5000, help='frequency of saving the latest results')
opt_gen.save_epoch_freq = 1 #type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
opt_gen.n_epochs = 100
opt_gen.isTrain = True
opt_gen.continue_train = False #action='store_true', help='continue training: load the latest model')
opt_gen.starting_epoch_count = opt_gen.parent_exp.starting_epoch_count #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_gen.phase = opt_gen.parent_exp.phase #type=str, default='train', help='train, val, test, etc')
opt_gen.name = 'rw_sims_gen' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
opt_gen.loss_type = "L2_sumL2"
opt_gen.niter = 100 #type=int, default=100, help='# of iter at starting learning rate')
opt_gen.niter_decay = 100 #type=int, default=100, help='# of iter to linearly decay learning rate to zero')



opt_gen.gpu_ids = opt_gen.parent_exp.gpu_ids #type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt_gen.num_threads = 4 #default=4, type=int, help='# threads for loading data')
opt_gen.checkpoints_load_dir =  opt_gen.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_gen.checkpoints_save_dir =  opt_gen.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_gen.results_dir = opt_gen.parent_exp.results_dir
opt_gen.log_dir =  opt_gen.parent_exp.log_dir #type=str, default='./checkpoints', help='models are saved here')
opt_gen.max_dataset_size = float("inf") #type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
opt_gen.verbose = False#action='store_true', help='if specified, print more debugging information')
opt_gen.suffix ='' #default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')