# RRDB
nf = 3
gc = 32

# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.0
lr = 10 ** log10_lr
lr3 = 10 ** -5.0
epochs = 50000
weight_decay = 1e-5
init_scale = 0.01

device_ids = [0]

# Super loss
lamda_reconstruction_1 = 2
lamda_reconstruction_2 = 2
lamda_guide_1 = 1
lamda_guide_2 = 1

lamda_low_frequency_1 = 1
lamda_low_frequency_2 = 1

use_imp_map = True
optim_step_1 = True
optim_step_2 = True
optim_step_3 = True

# Train:
batch_size = 24
cropsize = 128
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.98

# Val:
cropsize_val_coco = 256
cropsize_val_imagenet = 256
cropsize_val_div2k = 1024

batchsize_val = 3
shuffle_val = False
val_freq = 1

# Dataset
Dataset_mode = 'COCO'  # COCO / DIV2K /
Dataset_VAL_mode = 'DIV2K'  # COCO / DIV2K / ImageNet

TRAIN_PATH_DIV2K = '/media/disk2/jjp/jjp/Dataset/DIV2K/DIV2K_train_HR/'
VAL_PATH_DIV2K = '/media/disk2/jjp/jjp/Dataset/DIV2K/DIV2K_valid_HR/'

VAL_PATH_COCO = '/media/disk2/jjp/jjp/Dataset/COCO/val2017/'
TEST_PATH_COCO = '/media/disk2/jjp/jjp/Dataset/COCO/test2017/'

VAL_PATH_IMAGENET = '/media/data/jjp/Imagenet/ILSVRC2012_img_val'

# Display and logging:
loss_display_cutoff = 2.0  # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False

# Saving checkpoints:
MODEL_PATH = ''
checkpoint_on_error = True
SAVE_freq = 1


TEST_PATH = '/home/jjp/DeepMIH/image/'

TEST_PATH_cover = TEST_PATH + 'cover/'
TEST_PATH_secret_1 = TEST_PATH + 'secret_1/'
TEST_PATH_secret_2 = TEST_PATH + 'secret_2/'
TEST_PATH_steg_1 = TEST_PATH + 'steg_1/'
TEST_PATH_steg_2 = TEST_PATH + 'steg_2/'
TEST_PATH_secret_rev_1 = TEST_PATH + 'secret-rev_1/'
TEST_PATH_secret_rev_2 = TEST_PATH + 'secret-rev_2/'
TEST_PATH_imp_map = TEST_PATH + 'imp-map/'


# Load:
suffix_load = ''
tain_next = False

trained_epoch = 3000

pretrain = True
PRETRAIN_PATH = '/home/jjp/DeepMIH/model/'
suffix_pretrain = 'model_checkpoint_03000'
PRETRAIN_PATH_3 = '/home/jjp/DeepMIH/model/'
suffix_pretrain_3 = 'model_checkpoint_03000'