# vm_torch_intra_series_reg_mse_l2_dc_input_inference_new.py
# A script to perform intra-series image registration using a trained convolutional
# neural network

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 12th August 2022

# Import required module
import time

# Start timing experiment
exp_start = time.time()

# Import required modules
import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

# Import VoxelMorph with PyTorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import voxelmorphwip as vxmwip

# Path to folder containing folders of data files
dir_path = '/home/mr19/Documents/registration/data/all_frames_7_classes_npz'

# List of subject codes
subj_codes = ['aa', 'ah', 'gc', 'mr', 'br']

# List of data files per subject
n_file_list = []

# For each subject
for subj in subj_codes:

    # Find number of files and update n_files
    n_file_list.append(len(glob.glob(os.path.join(dir_path, subj, '*.npz'))))

# Preallocate NumPy arrays for images and corresponding segmentations
img_array = np.zeros((sum(n_file_list), 256, 256))
seg_array = np.zeros((sum(n_file_list), 256, 256))

# Preallocate array for subject codes
subj_array = []

# Counter for number of files
n_files = 0

# For each subject
for jdx, subj in enumerate(subj_codes):

    # Add list of subject code to subj_array
    subj_array += ([subj] * n_file_list[jdx])

    # For each data file
    for idx in range(n_file_list[jdx]):

        # Load each image and segmentation
        tmp_data = np.load(os.path.join(dir_path, subj, f'10fps_{subj}_frame_{idx + 1}.npz'))

        # Populate img_array and seg_array
        img_array[(n_files + idx), ...] = tmp_data['img']
        seg_array[(n_files + idx), ...] = tmp_data['seg']
    
    # Update counter
    n_files += n_file_list[jdx]

# Convert subj_array from list to NumPy array
subj_array = np.asarray(subj_array)

# Path to CSV file listing paths to optimal segmentation networks
opt_seg_net_paths = '/home/mr19/Documents/registration/voxelmorph/opt_seg_net_paths.csv'

# Load CSV file
opt_seg_net_paths = pd.read_csv(opt_seg_net_paths)

# Import required modules
from torch import is_tensor
from torch.utils.data import Dataset
from numpy.random import randint

# Create a custom dataset
class SpeechMRIData(Dataset):
    
    # Dataset of two MR images (moving and fixed image), the predicted and ground-truth segmentations
    # of these images
    def __init__(self, img_array, seg_array, gt_array, subj_array, transform=None):
        
        # Args:
        # 1) img_array (NumPy array): NumPy array of images.
        # 2) seg_array (NumPy array): Numpy array of predicted segmentations.
        # 3) gt_array (Numpy array): Numpy array of ground-truth segmentations.
        # 4) subj_array (Numpy array): Numpy array of subject codes.
        # 5) transform (callable, optional): Optional transform to be applied on a sample.

        # Update self
        self.img_array = img_array
        self.seg_array = seg_array
        self.gt_array = gt_array
        self.subj_array = subj_array
        self.transform = transform

    def __len__(self):
        return len(self.subj_array)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        # Load fixed image and corresponding predicted and ground-truth segmentations
        fix_img = self.img_array[idx, ...]
        fix_seg = self.seg_array[idx, ...]
        fix_gt = self.gt_array[idx, ...]

        # Identify subject
        subj_code = self.subj_array[idx]
        
        # Extract images of subject and corresponding predicted and ground-truth segmentations
        subj_img = self.img_array[self.subj_array == subj_code, ...]
        subj_seg = self.seg_array[self.subj_array == subj_code, ...]
        subj_gt = self.gt_array[self.subj_array == subj_code, ...]
                
        # Extract specific image of subject and corresponding predicted segmentations
        # (specific image depends on subject)
        if subj_code == 'gc':
            
            mov_img = subj_img[1, ...]
            mov_seg = subj_seg[1, ...]
            mov_gt = subj_gt[1, ...]

        else:

            mov_img = subj_img[0, ...]
            mov_seg = subj_seg[0, ...]
            mov_gt = subj_gt[0, ...]

        # Define sample
        sample = {'mov_img': mov_img, 'mov_seg': mov_seg, 'mov_gt': mov_gt, 'fix_img': fix_img, 'fix_seg': fix_seg, 'fix_gt': fix_gt}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

# Import required modules
from numpy import int64, newaxis
from torch import from_numpy
from torch.nn.functional import one_hot

# Create a transform
class ToTensor(object):
    
    # To convert numpy arrays in a sample to tensors

    def __call__(self, sample):
        mov_img, mov_seg, mov_gt, fix_img, fix_seg, fix_gt = sample['mov_img'], sample['mov_seg'], sample['mov_gt'], sample['fix_img'], sample['fix_seg'], sample['fix_gt']

        # Add channel to mov_img and fix_img and convert to correct data type
        mov_img = from_numpy(mov_img[newaxis, :, :]).float()
        fix_img = from_numpy(fix_img[newaxis, :, :]).float()

        # One-hot encode mov_seg, mov_gt, fix_seg and fix_gt, and then permute dimensions and convert to 
        # correct data type
        mov_seg = one_hot(from_numpy(int64(mov_seg))).permute(2, 0, 1).float()
        mov_gt = one_hot(from_numpy(int64(mov_gt))).permute(2, 0, 1).float()
        fix_seg = one_hot(from_numpy(int64(fix_seg))).permute(2, 0, 1).float()
        fix_gt = one_hot(from_numpy(int64(fix_gt))).permute(2, 0, 1).float()

        return {'mov_img': mov_img, 'mov_seg': mov_seg, 'mov_gt': mov_gt, 'fix_img': fix_img, 'fix_seg': fix_seg, 'fix_gt': fix_gt}

# Define Dice coefficient
class EachClassDC:
    """
    Dice coefficient of each class
    """

    def loss(self, y_true, y_pred):
        
        # One-hot encode PyTorch tensors
        y_true = one_hot(y_true)
        y_pred = one_hot(y_pred) 
        top = 2 * (y_true * y_pred).sum(dim = (1, 2))
        bottom = torch.clamp((y_true + y_pred).sum(dim = (1, 2)), min = 1e-5)
        return top / bottom

# Import required modules
from torchvision import transforms

# Function to create evaluating datasets
def eval_ds_fn(eval_subj_code, img_array, seg_array, gt_array, subj_array):

    # Create lists of locations of evaluating data
    eval_img = img_array[subj_array == eval_subj_code, ...]
    eval_seg = seg_array[subj_array == eval_subj_code, ...]
    eval_gt = gt_array[subj_array == eval_subj_code, ...]
    eval_subj = subj_array[subj_array == eval_subj_code]

    # Define transformation
    data_transform = transforms.Compose([ToTensor()])

    # Create evaluating dataset
    eval_ds = SpeechMRIData(eval_img, eval_seg, eval_gt, eval_subj, transform = data_transform)

    return eval_ds

# Function to evaluate the network
def evaluate(model, device, eval_loader, inshape, losses, loss_weights):
    
    # Set network to evaluation mode
    model.eval()

    # Preallocate array for losses and Dice coefficients
    eval_losses = np.zeros((len(eval_loader), len(losses)))
    dice_coeffs = np.zeros(len(eval_loader))
    dice_coeffs_pred = np.zeros(len(eval_loader))

    # Without tracking gradients
    with torch.no_grad():

        # For each minibatch
        for batch_idx, eval_data in enumerate(eval_loader):

            # Extract images and segmentations and then send them to GPU
            mov_img = eval_data['mov_img'].to(device)
            mov_seg = eval_data['mov_seg'].to(device)
            mov_gt = eval_data['mov_gt'].to(device)
            fix_img = eval_data['fix_img'].to(device)
            fix_seg = eval_data['fix_seg'].to(device)
            fix_gt = eval_data['fix_gt'].to(device)

            # Create tensor of zeros (required for deformation field)
            zero_phi = torch.cat((torch.zeros_like(mov_img), torch.zeros_like(mov_img)), 1).to(device)

            # Create required lists
            input = [torch.cat((mov_img, mov_seg[:, 1:, ...]), 1), torch.cat((fix_img, fix_seg[:, 1:, ...]), 1), mov_gt[:, 1:, ...]]
            y_true = [torch.cat((fix_img, fix_gt[:, 1:, ...]), 1), zero_phi]

            # Forward propagation
            y_pred = model(*input)
        
            # Create coordinate tensor
            coord_tensor = torch.arange(1, 256*256 + 1, dtype=torch.float) / 100000
            coord_tensor = torch.reshape(coord_tensor, (256, -1))
            coord_tensor = torch.unsqueeze(coord_tensor, 0)
            coord_tensor = torch.unsqueeze(coord_tensor, 0).repeat(mov_img.shape[0], 1, 1, 1)
            coord_tensor = coord_tensor.to(device)

            # Create spatial transformer
            s_transformer = vxm.layers.SpatialTransformer(inshape, mode = 'nearest')

            # Move spatial transformer to GPU
            s_transformer.to(device)

            # Transform coordinate tensor
            tra_coord_tensor = s_transformer(coord_tensor, y_pred[1]) * 100000

            # Transform segmentations
            tra_mov_seg = s_transformer(mov_seg, y_pred[1])

        return tra_coord_tensor[:, 0, ...], y_pred[1], torch.argmax(tra_mov_seg, dim = 1)

# Specify experiment number
exp_n = 4

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Minibatch size
mb_size = 4

# Number of epochs
n_epochs = 200

# Data augmentation
aug_ds = 4

# U-Net architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]
inshape = img_array.shape[1:]
nb_features = [enc_nf, dec_nf]

# Losses in loss function
losses = [vxm.losses.MSE().loss, vxmwip.losses.Grad('l2').loss, vxm.losses.Dice().loss]
loss_weights = [1, 1, 1]

# Path to CSV file of optimal hyperparameters
opt_hp_path = f'/home/mr19/Documents/registration/voxelmorph/mse_l2_dc_aug_{aug_ds}_experiment_{exp_n}/mse_l2_dc_aug_{aug_ds}_experiment_{exp_n}_opt_hp_v_c.csv'

# Load optimal hyperparameters
opt_hp_df = pd.read_csv(opt_hp_path, dtype = str)

# Convert subj_codes to NumPy array
subj_codes = np.asarray(subj_codes)

# For each subject
for fold, test_subj in enumerate(subj_codes):

    # Create array for predicted segmentations
    tmp_seg_array = np.copy(seg_array)

    # For each subject
    for tmp_subj in subj_codes:

        # Path to predicted segmentations
        pred_seg_path = os.path.join('/home/mr19/Documents/segmentation_2d_results', opt_seg_net_paths[f'test_{tmp_subj}'][0], opt_seg_net_paths[f'test_{tmp_subj}'][0] + '_eval_predict_cca.mat')

        # Load predicted segmentations
        tmp_seg_array[subj_array == tmp_subj, ...] = np.float64(np.transpose(sio.loadmat(pred_seg_path)['cca_predict'], (2, 0, 1)))

    # Print summary
    print(f'Testing subject: {test_subj}   Experiment: {exp_n}')

    # Create dataset for evaluation
    eval_ds = eval_ds_fn(test_subj, img_array, tmp_seg_array, seg_array, subj_array)

    # Extract learning rate, lambda and gamma
    l_rate = opt_hp_df.set_index('test_subj').at[test_subj, 'l_rate']
    lambda_param = opt_hp_df.set_index('test_subj').at[test_subj, 'l_val']
    gamma_param = opt_hp_df.set_index('test_subj').at[test_subj, 'g_val']

    # If gamma_param is equal to 1, change to integer
    if gamma_param == 1:
        gamma_param == 1
        
    # Print summary
    print(f'Learning rate: {l_rate}   Lambda: {lambda_param}   Gamma: {gamma_param}')

    # Create batches of evaluation data               
    allevaldataloader = DataLoader(eval_ds, batch_size = len(eval_ds),
                        shuffle = False, num_workers = 4)

    # Create U-Net model using VxmDense
    vxm_model = vxmwip.networks.VxmDense(inshape, nb_features, int_steps=0)

    # Zero gradient buffers of model
    vxm_model.zero_grad() 

    # Assemble name of folder containing model parameters
    model_param_dir = f'test_{test_subj}_lr_{l_rate}_mbs_{mb_size}_epochs_{n_epochs}_mse_1_l2_{lambda_param}_dc_{gamma_param}_aug_{aug_ds}'

    # Assemble path to model parameters
    model_param_path = os.path.join(f'/home/mr19/Documents/registration/voxelmorph/mse_l2_dc_aug_{aug_ds}_experiment_{exp_n}', model_param_dir, model_param_dir + '_parameters.pth')

    # Load model parameters
    vxm_model.load_state_dict(torch.load(model_param_path))

    # Send model to GPU
    vxm_model.to(device)

    # Evaluate network using evaluation dataset
    eval_tra_coords, eval_disp_field, eval_tra_est_mov_seg = evaluate(vxm_model, device, allevaldataloader, inshape, losses, loss_weights)

    # Specify folder where all files will be saved
    save_dir = f'test_{test_subj}_lr_{l_rate}_mbs_{mb_size}_epochs_{n_epochs}_mse_1_l2_{lambda_param}_dc_{gamma_param}_aug_{aug_ds}'
    save_path = os.path.join(f'/home/mr19/Documents/registration/voxelmorph/mse_l2_dc_aug_{aug_ds}_experiment_{exp_n}', save_dir)

    # Check if folder exists and create if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Specify estimated displacement field file name and path
    disp_field_name = save_dir + '_eval_disp_field.mat'
    disp_field_name = os.path.join(save_path, disp_field_name)

    # Save estimates as MAT file
    sio.savemat(disp_field_name, {'eval_disp_field': eval_disp_field.cpu().numpy()})

    # Specify transformed coordinate file name and path
    tra_coords_name = save_dir + '_eval_tra_coords.mat'
    tra_coords_name = os.path.join(save_path, tra_coords_name)

    # Save transformed coordinates as MAT file
    sio.savemat(tra_coords_name, {'eval_tra_coords': np.transpose(eval_tra_coords.cpu().numpy(), (1, 2, 0))})

    # Specify transformed estimated moving segmentations file name and path
    tra_est_mov_seg_name = save_dir + '_eval_tra_est_mov_seg.mat'
    tra_est_mov_seg_name = os.path.join(save_path, tra_est_mov_seg_name)

    # Save transformed estimated segmentations as MAT file
    sio.savemat(tra_est_mov_seg_name, {'eval_tra_est_mov_seg': np.transpose(eval_tra_est_mov_seg.cpu().numpy(), (1, 2, 0))})

    # Stop timing experiment
    exp_end = time.time()
    exp_min, exp_sec = divmod(exp_end - exp_start, 60)
    exp_hou, exp_min = divmod(exp_min, 60)
    print('Time elapsed in hh:mm:ss: {:02.0f}:{:02.0f}:{:02.0f}'.format(exp_hou, exp_min, exp_sec))