# vm_torch_intra_series_reg_mse_l2_dc_input_new.py
# A script to perform intra series image registration using VoxelMorph
# with a PyTorch backend

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 8th September 2022

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
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader

# Import required classes
from DataAugmentation import ToTensor, RotateCropAndPad, RandomCrop, Rescale, RandomTranslation, RescaleAndPad
from SpeechMRIDatasets import SpeechMRIData, SpeechMRIDataRandom

# Import VoxelMorph with PyTorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import voxelmorphwip as vxmwip


def create_dataset(data_dir, subj_id_list, augmentation=True, random_choice=True):
    """Function to create a dataset. 

    Args:
        - data_dir (Path): path to folder containing all data (i.e. images
          and ground-truth segmentations)
        - subj_id_list (list of integers): list of IDs of subjects 
          whose images should be included in dataset.
          For example, subj_id_list = [1,2,3,4] would indicate
          that the images of subjects 1, 2, 3 and 4 should be included
          in the dataset
        - augmentation (True or False): indicates if data should be 
          augmented using rotations, translations, cropping and rescaling
        - random_choice (True or False): indicates if SpeechMRIData or 
          SpeechMRIDataRandom should be used as the Dataset class
        
    Returns:
        
        - PyTorch dataset
        - loss_weighting (PyTorch tensor): the weighting of each class in 
          the loss function
    """
    
    # Preallocate list for paths to images and corresponding 
    # segmentations (ground-truths and estimated) to include in dataset
    full_img_list = []
    full_seg_list = []
    full_gt_list = []
    full_subj_id_list = []

    # For each subject ID
    for subj_id in subj_id_list:

        # Create lists of frames in the subfolders
        img_list = [int(g[6:-4]) for g in os.listdir(data_dir / 'Normalised_Images' / f'Subject{subj_id}') if (g.endswith('.mat') and g.startswith('image'))]
        img_list.sort()
        gt_list = [int(g[5:-4]) for g in os.listdir(data_dir / 'GT_Segmentations' / f'Subject{subj_id}') if (g.endswith('.mat') and g.startswith('mask'))]
        gt_list.sort()
        seg_list = [int(g[5:-4]) for g in os.listdir(data_dir / 'Est_Segmentations' / f'Subject{subj_id}') if (g.endswith('.mat') and g.startswith('mask'))]
        seg_list.sort()

        # Print update
        print(f'Subject {subj_id} has {len(img_list)} images and corresponding segmentations')

        # Create lists of paths to images and segmentations
        img_list = [data_dir / 'Normalised_Images' / f'Subject{subj_id}' / f'image_{g}.mat' for g in img_list]
        gt_list = [data_dir / 'GT_Segmentations' / f'Subject{subj_id}' / f'mask_{g}.mat' for g in gt_list]
        seg_list = [data_dir / 'Est_Segmentations' / f'Subject{subj_id}' / f'mask_{g}.mat' for g in seg_list]

        # Update full_img_list, full_gt_list, full_seg_list and full_subj_id_list
        full_img_list += img_list
        full_gt_list += gt_list
        full_seg_list += seg_list
        full_subj_id_list += ([subj_id] * len(img_list))

    # Load first image
    frame = sio.loadmat(full_img_list[0])['image_frame']

    # Image dimensions
    first_frame_dim = frame.shape

    # Preallocate NumPy arrays for images and segmentations
    img_array = np.zeros((len(full_img_list), first_frame_dim[0], first_frame_dim[1]))
    gt_array = np.zeros((len(full_img_list), first_frame_dim[0], first_frame_dim[1]))
    seg_array = np.zeros((len(full_img_list), first_frame_dim[0], first_frame_dim[1]))

    # Create NumPy array of subject IDs
    subj_id_array = np.asarray(full_subj_id_list)

    # For each image
    for idx, file_path in enumerate(full_img_list):

        # Populate img_array
        img_array[idx, ...] = sio.loadmat(file_path)['image_frame']

    # For each GT segmentation
    for idx, file_path in enumerate(full_gt_list):

        # Populate gt_array
        gt_array[idx, ...] = sio.loadmat(file_path)['mask_frame'].astype('int64')

    for idx, file_path in enumerate(full_seg_list):

        # Populate seg_array
        seg_array[idx, ...] = sio.loadmat(file_path)['mask_frame'].astype('int64')

    # If required, augment the dataset
    if augmentation:

        # Create dataset with transformation
        data_transform = transforms.Compose([ToTensor()])
        dataset_1 = SpeechMRIDataRandom(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

        # Create dataset with transformation
        data_transform = transforms.Compose([RotateCropAndPad((-30, 10)), ToTensor()])
        dataset_2 = SpeechMRIDataRandom(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

        # Create dataset with transformation
        data_transform = transforms.Compose([RandomCrop((15, 30), (5, 10), 220), Rescale(256), ToTensor()])
        dataset_3 = SpeechMRIDataRandom(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

        # Create dataset with transformation 
        data_transform = transforms.Compose([RandomTranslation((-30, 30), (0, 30)), ToTensor()])
        dataset_4 = SpeechMRIDataRandom(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

        # Create dataset with transformation 
        data_transform = transforms.Compose([RescaleAndPad((210, 255)), ToTensor()])
        dataset_5 = SpeechMRIDataRandom(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

        # Combine datasets
        dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, dataset_5])

    elif random_choice:
        
        # Define transformation
        data_transform = transforms.Compose([ToTensor()])

        # Create training, validating and testing datasets
        dataset = SpeechMRIDataRandom(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

    else:

        # Define transformation
        data_transform = transforms.Compose([ToTensor()])

        # Create training, validating and testing datasets
        dataset = SpeechMRIData(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

    return dataset

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


# Define multi-class Dice coefficient
class MCDC:
    """
    Multi-class Dice coefficient
    """

    def loss(self, y_true, y_pred):
        top = 2 * (y_true * y_pred).sum(dim = (0, 2, 3))
        bottom = torch.clamp((y_true + y_pred).sum(dim = (0, 2, 3)), min = 1e-5)
        dice = torch.mean(top / bottom)
        return dice


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
    

# Function to create training datasets
def train_ds_fn(eval_subj_code, img_array, seg_array, gt_array, subj_array):

    # Create lists of locations of training datasets
    train_img = img_array[subj_array != eval_subj_code, ...]
    train_seg = seg_array[subj_array != eval_subj_code, ...]
    train_gt = gt_array[subj_array != eval_subj_code, ...]
    train_subj = subj_array[subj_array != eval_subj_code]

    # Define transformation
    data_transform = transforms.Compose([ToTensor()])

    # Create training dataset
    train_ds_1 = SpeechMRIDataRandom(train_img, train_seg, train_gt, train_subj, transform = data_transform)                                                              

    # Define transformation
    data_transform = transforms.Compose([RotateCropAndPad((-20, 10)), ToTensor()])

    # Create training dataset
    train_ds_2 = SpeechMRIDataRandom(train_img, train_seg, train_gt, train_subj, transform = data_transform)

    # Define transformation
    data_transform = transforms.Compose([RandomCrop((15, 30), (5, 10), 220), Rescale(256), ToTensor()])

    # Create training dataset
    train_ds_3 = SpeechMRIDataRandom(train_img, train_seg, train_gt, train_subj, transform = data_transform)

    # Define transformation
    data_transform = transforms.Compose([RandomTranslation((-30, 30), (0, 30)), ToTensor()])

    # Create training dataset
    train_ds_4 = SpeechMRIDataRandom(train_img, train_seg, train_gt, train_subj, transform = data_transform)

    # Define transformation
    data_transform = transforms.Compose([RescaleAndPad((210, 255)), ToTensor()])

    # Create training dataset
    train_ds_5 = SpeechMRIDataRandom(train_img, train_seg, train_gt, train_subj, transform = data_transform)

    # Combine training datasets
    training_dataset = ConcatDataset([train_ds_1, train_ds_2, train_ds_3, train_ds_4, train_ds_5])
    # training_dataset = train_ds_1

    # Number of augmented datasets
    aug_ds = floor(len(training_dataset) / len(train_ds_1) - 1)

    return training_dataset, aug_ds

# Function to create training and evaluating datasets
def train_eval_ds_fn(eval_subj_code, img_array, seg_array, gt_array, subj_array):

    # Create lists of locations of training and evaluating datasets
    train_img = img_array[subj_array != eval_subj_code, ...]
    train_seg = seg_array[subj_array != eval_subj_code, ...]
    train_gt = gt_array[subj_array != eval_subj_code, ...]
    train_subj = subj_array[subj_array != eval_subj_code]
    eval_img = img_array[subj_array == eval_subj_code, ...]
    eval_seg = seg_array[subj_array == eval_subj_code, ...]
    eval_gt = gt_array[subj_array == eval_subj_code, ...]
    eval_subj = subj_array[subj_array == eval_subj_code]

    # Define transformation
    data_transform = transforms.Compose([ToTensor()])

    # Create training and evaluating datasets
    train_ds = SpeechMRIData(train_img, train_seg, train_gt, train_subj, transform = data_transform)
    eval_ds = SpeechMRIData(eval_img, eval_seg, eval_gt, eval_subj, transform = data_transform)

    return train_ds, eval_ds

# Function to train the network
def train(model, device, train_loader, optimizer, losses, loss_weights):
    
    # Set network to training mode
    model.train()

    # For each minibatch
    for batch_idx, train_data in enumerate(train_loader):
        
        # Extract images and segmentations and then send them to GPU
        mov_img = train_data['mov_img'].to(device)
        mov_seg = train_data['mov_seg'].to(device)
        mov_gt = train_data['mov_gt'].to(device)
        fix_img = train_data['fix_img'].to(device)
        fix_seg = train_data['fix_seg'].to(device)
        fix_gt = train_data['fix_gt'].to(device)

        # Create tensor of zeros (required for deformation field)
        zero_phi = torch.cat((torch.zeros_like(mov_img), torch.zeros_like(mov_img)), 1).to(device)

        # Create required lists
        input = [torch.cat((mov_img, mov_seg[:, 1:, ...]), 1), torch.cat((fix_img, fix_seg[:, 1:, ...]), 1), mov_gt[:, 1:, ...]]
        y_true = [torch.cat((fix_img, fix_gt[:, 1:, ...]), 1), zero_phi]

        # Zero the optimiser gradients
        optimizer.zero_grad()

        # Forward propagation
        y_pred = model(*input)
        
        # Calculate loss
        loss = 0
        curr_loss = losses[0](y_true[0][:, 0, ...], y_pred[0][:, 0, ...]) * loss_weights[0]
        loss += curr_loss

        # Calculate loss
        curr_loss = losses[1](y_true[1], y_pred[1]) * loss_weights[1]
        loss += curr_loss

        # Calculate loss
        curr_loss = losses[2](y_true[0][:, 1:, ...], y_pred[0][:, 1:, ...]) * loss_weights[2]
        loss += curr_loss

        # Backwards propagation
        loss.backward()

        # Optimise weights
        optimizer.step()

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

            # Calculate loss
            loss = 0
            curr_loss = losses[0](y_true[0][:, 0, ...], y_pred[0][:, 0, ...]) * loss_weights[0]
            loss += curr_loss

            # Populate eval_losses
            eval_losses[batch_idx, 0] = curr_loss.item()

            # Calculate loss
            curr_loss = losses[1](y_true[1], y_pred[1]) * loss_weights[1]
            loss += curr_loss

            # Populate eval_losses
            eval_losses[batch_idx, 1] = curr_loss.item()

            # Calculate loss
            curr_loss = losses[2](y_true[0][:, 1:, ...], y_pred[0][:, 1:, ...]) * loss_weights[2]
            loss += curr_loss

            # Populate eval_losses
            eval_losses[batch_idx, 2] = curr_loss.item()
        
            # Create spatial transformer
            s_transformer = vxm.layers.SpatialTransformer(inshape, mode = 'nearest')

            # Move spatial transformer to GPU
            s_transformer.to(device)

            # Transform segmentations
            tra_mov_gt = s_transformer(mov_gt, y_pred[1])

            # Calculate multi-class Dice coefficient and populate dice_coeffs and dice_coeffs_pred
            dice_coeffs[batch_idx] = MCDC().loss(y_true[0][:, 1:, ...], tra_mov_gt[:, 1:, ...]).item()
            dice_coeffs_pred[batch_idx] = MCDC().loss(fix_seg[:, 1:, ...], tra_mov_gt[:, 1:, ...]).item()

        return eval_losses, y_pred[0][:, 0, ...], dice_coeffs, dice_coeffs_pred, torch.argmax(tra_mov_gt, dim = 1)

# Specify experiment number
exp_n = 4

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# List of learning rates
lr_array = [0.0003] * 4 + [0.0009] * 4
lr_array = [0.0009, 0.0009, 0.0009, 0.0009, 0.0009]

# Minibatch size
mb_size = 4

# Number of epochs
n_epochs = 200

# U-Net architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]
inshape = img_array.shape[1:]
nb_features = [enc_nf, dec_nf]

# Losses in loss function
losses = [vxm.losses.MSE().loss, vxmwip.losses.Grad('l2').loss, vxm.losses.Dice().loss]

# Array of lambda values
lambda_array = [0.001] * 2 + [0.01] * 2
lambda_array = lambda_array * 2
lambda_array = [0.01, 0.01, 0.01, 0.001, 0.001]

# Array of gamma values
gamma_array = [0.1, 1] * 4
gamma_array = [1, 0.1, 1, 1, 0.1]

# Convert subj_codes to NumPy array
subj_codes = np.asarray(subj_codes)

# Hyperparameter optimisation
hp_opt = 0

# For each subject
for fold, test_subj in enumerate(subj_codes):

    # If there is hyperparameter optimisation
    if hp_opt == 1:
        
        # Array of codes of remaining subjects
        rem_subj = subj_codes[subj_codes != test_subj]

        # Create array for predicted segmentations
        tmp_seg_array = np.copy(seg_array)

        # For each remaining subject
        for val_subj in rem_subj:

            # Path to predicted segmentations
            pred_seg_path = os.path.join('/home/mr19/Documents/segmentation_2d_results', opt_seg_net_paths[f'test_{test_subj}_val_{val_subj}'][0], opt_seg_net_paths[f'test_{test_subj}_val_{val_subj}'][0] + '_eval_predict_cca.mat')

            # Load predicted segmentations
            tmp_seg_array[subj_array == val_subj, ...] = np.float64(np.transpose(sio.loadmat(pred_seg_path)['cca_predict'], (2, 0, 1)))

        # For each remaining subject
        for val_subj in rem_subj:

            # Print summary
            print(f'Testing subject: {test_subj}   Validating subject: {val_subj}   Experiment: {exp_n}')

            # Create training dataset
            train_ds, aug_ds = train_ds_fn(val_subj,
                                            img_array[subj_array != test_subj, ...],
                                            tmp_seg_array[subj_array != test_subj, ...],
                                            seg_array[subj_array != test_subj, ...],
                                            subj_array[subj_array != test_subj])
    
            # Create other training dataset and evaluating dataset
            other_train_ds, eval_ds = train_eval_ds_fn(val_subj,
                                                        img_array[subj_array != test_subj, ...],
                                                        tmp_seg_array[subj_array != test_subj, ...],
                                                        seg_array[subj_array != test_subj, ...],
                                                        subj_array[subj_array != test_subj])

            # Create batches of training and evaluating data
            traindataloader = DataLoader(train_ds, batch_size = mb_size,
                                    shuffle = True, num_workers = 4)
            alltraindataloader = DataLoader(other_train_ds, batch_size = len(other_train_ds),
                                    shuffle = False, num_workers = 4)
            singletraindataloader = DataLoader(other_train_ds, batch_size = 1,
                                    shuffle = False, num_workers = 4)                     
            allevaldataloader = DataLoader(eval_ds, batch_size = len(eval_ds),
                                    shuffle = False, num_workers = 4)
            singleevaldataloader = DataLoader(eval_ds, batch_size = 1,
                                    shuffle = False, num_workers = 4)

            # For each learning rate and lambda combination
            for hyp_comb_idx in range(len(lr_array)):

                # Extract learning rate, lambda and gamma
                l_rate = lr_array[hyp_comb_idx]
                lambda_param = lambda_array[hyp_comb_idx]
                gamma_param = gamma_array[hyp_comb_idx]
                
                # Print summary
                print(f'Learning rate: {l_rate}   Lambda: {lambda_param}   Gamma: {gamma_param}')

                # Create U-Net model using VxmDense
                vxm_model = vxmwip.networks.VxmDense(inshape, nb_features, int_steps=0)

                # Zero gradient buffers of model
                vxm_model.zero_grad() 

                # Send model to GPU
                vxm_model.to(device)

                # Create the optimizer
                optimizer = torch.optim.Adam(vxm_model.parameters(), lr = l_rate)

                # Weighting for each loss
                loss_weights = [1, lambda_param, gamma_param]
                
                # Preallocate arrays for losses
                all_train_losses_array = np.empty([n_epochs, len(losses)])
                all_eval_losses_array = np.empty([n_epochs, len(losses)])
                single_train_losses_array = np.empty([len(singletraindataloader), n_epochs * len(losses)])
                single_eval_losses_array = np.empty([len(singleevaldataloader), n_epochs * len(losses)])

                # Preallocate arrays for Dice coefficients
                all_train_dc_array = np.empty(n_epochs)
                all_train_dc_pred_array = np.empty(n_epochs)
                all_eval_dc_array = np.empty(n_epochs)
                all_eval_dc_pred_array = np.empty(n_epochs)
                single_train_dc_array = np.empty([len(singletraindataloader), n_epochs])
                single_train_dc_pred_array = np.empty([len(singletraindataloader), n_epochs])
                single_eval_dc_array = np.empty([len(singleevaldataloader), n_epochs])
                single_eval_dc_pred_array = np.empty([len(singleevaldataloader), n_epochs])

                # Train and evaluate network

                # For each epoch
                for epoch in range(n_epochs):
                    
                    # Measure time
                    start_time = time.time()

                    # Train network
                    train(vxm_model, device, traindataloader, optimizer, losses, loss_weights)

                    # Evaluate network using training dataset
                    train_losses, train_pred, train_dc, train_dc_pred, train_seg = evaluate(vxm_model, device, singletraindataloader, inshape, losses, loss_weights)

                    # Populate single_train_losses_array
                    single_train_losses_array[:, (len(losses) * epoch):(len(losses) * (epoch + 1))] = train_losses

                    # Populate single_train_dc_array and single_train_dc_pred_array
                    single_train_dc_array[:, epoch] = train_dc
                    single_train_dc_pred_array[:, epoch] = train_dc_pred

                    # Evaluate network using training dataset
                    train_losses, train_pred, train_dc, train_dc_pred, train_seg = evaluate(vxm_model, device, alltraindataloader, inshape, losses, loss_weights)

                    # Populate all_train_losses_array
                    all_train_losses_array[epoch, :] = train_losses

                    # Populate all_train_dc_array and all_train_dc_pred_array
                    all_train_dc_array[epoch] = train_dc
                    all_train_dc_pred_array[epoch] = train_dc_pred

                    # Evaluate network using evaluating dataset
                    eval_losses, eval_pred, eval_dc, eval_dc_pred, eval_seg = evaluate(vxm_model, device, singleevaldataloader, inshape, losses, loss_weights)

                    # Populate single_eval_losses_array
                    single_eval_losses_array[:, (len(losses) * epoch):(len(losses) * (epoch + 1))] = eval_losses

                    # Populate single_eval_dc_array and single_eval_dc_pred_array
                    single_eval_dc_array[:, epoch] = eval_dc
                    single_eval_dc_pred_array[:, epoch] = eval_dc_pred

                    # Evaluate network using evaluating dataset
                    eval_losses, eval_pred, eval_dc, eval_dc_pred, eval_seg = evaluate(vxm_model, device, allevaldataloader, inshape, losses, loss_weights)

                    # Populate all_eval_losses_array
                    all_eval_losses_array[epoch, :] = eval_losses

                    # Populate all_eval_dc_array and all_eval_dc_pred_array
                    all_eval_dc_array[epoch] = eval_dc
                    all_eval_dc_pred_array[epoch] = eval_dc_pred

                    # Print epoch and loss information
                    print('Epoch: %d  Time: %.2f sec\nTraining loss: %.5f, %.5f, %.5f   Evaluating loss: %.5f, %.5f, %.5f\nTraining MCDC: %.2f   Evaluating MCDC: %.2f' %
                            (epoch + 1, time.time() - start_time,
                            train_losses[:, 0], train_losses[:, 1], train_losses[:, 2],
                            eval_losses[:, 0], eval_losses[:, 1], eval_losses[:, 2],
                            train_dc, eval_dc))

                print('Finished Training')

                # Specify folder where all files will be saved
                save_dir = f'test_{test_subj}_val_{val_subj}_lr_{l_rate}_mbs_{mb_size}_epochs_{n_epochs}_mse_1_l2_{lambda_param}_dc_{gamma_param}_aug_{aug_ds}'
                save_path = os.path.join(f'/home/mr19/Documents/registration/voxelmorph/mse_l2_dc_aug_{aug_ds}_experiment_{exp_n}', save_dir)

                # Check if folder exists and create if necessary
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # Specify network name and path
                network_name = save_dir + '_parameters.pth'
                network_name = os.path.join(save_path, network_name)

                # Save network parameters
                torch.save(vxm_model.state_dict(), network_name)

                # Specify losses file name and path
                losses_name = save_dir + '_single_train_loss.csv'
                losses_name = os.path.join(save_path, losses_name)

                # Save losses array in a csv file
                np.savetxt(losses_name, single_train_losses_array, delimiter = ',')

                # Preallocate pandas DataFrame for losses and Dice coefficients
                df = pd.DataFrame()

                # Create pandas DataFrame of training losses
                tmp_df = pd.DataFrame(all_train_losses_array, columns=['mse_tr', 'gl_tr', 'adc_tr'])
                
                # Calculate total training loss
                tmp_df['total_tr'] = tmp_df.sum(axis=1)

                # Populate df
                df = pd.concat([df, tmp_df], axis=1)

                # Specify losses file name and path
                losses_name = save_dir + '_single_eval_loss.csv'
                losses_name = os.path.join(save_path, losses_name)

                # Save losses array in a csv file
                np.savetxt(losses_name, single_eval_losses_array, delimiter = ',')

                # Create pandas DataFrame of evaluating losses
                tmp_df = pd.DataFrame(all_eval_losses_array, columns=['mse_ev', 'gl_ev', 'adc_ev'])
                
                # Calculate total evaluating loss
                tmp_df['total_ev'] = tmp_df.sum(axis=1)

                # Populate df
                df = pd.concat([df, tmp_df], axis=1)

                # Specify Dice coefficient file name and path
                dc_name = save_dir + '_single_train_dc_gt.csv'
                dc_name = os.path.join(save_path, dc_name)

                # Save Dice coefficient array in a csv file
                np.savetxt(dc_name, single_train_dc_array, delimiter = ',')

                # Specify Dice coefficient file name and path
                dc_name = save_dir + '_single_train_dc_pred.csv'
                dc_name = os.path.join(save_path, dc_name)

                # Save Dice coefficient array in a csv file
                np.savetxt(dc_name, single_train_dc_pred_array, delimiter = ',')

                # Create pandas DataFrame of mean Dice coefficient
                tmp_df = pd.DataFrame(all_train_dc_array, columns=['dc_gt_tr'])

                # Populate df
                df = pd.concat([df, tmp_df], axis=1)

                # Create pandas DataFrame of mean Dice coefficient
                tmp_df = pd.DataFrame(all_train_dc_pred_array, columns=['dc_pred_tr'])

                # Populate df
                df = pd.concat([df, tmp_df], axis=1)

                # Specify Dice coefficient file name and path
                dc_name = save_dir + '_single_eval_dc_gt.csv'
                dc_name = os.path.join(save_path, dc_name)

                # Save Dice coefficient array in a csv file
                np.savetxt(dc_name, single_eval_dc_array, delimiter = ',')

                # Specify Dice coefficient file name and path
                dc_name = save_dir + '_single_eval_dc_pred.csv'
                dc_name = os.path.join(save_path, dc_name)

                # Save Dice coefficient array in a csv file
                np.savetxt(dc_name, single_eval_dc_pred_array, delimiter = ',')

                # Create pandas DataFrame of mean Dice coefficient
                tmp_df = pd.DataFrame(all_eval_dc_array, columns=['dc_gt_ev'])

                # Populate df
                df = pd.concat([df, tmp_df], axis=1)

                # Create pandas DataFrame of mean Dice coefficient
                tmp_df = pd.DataFrame(all_eval_dc_pred_array, columns=['dc_pred_ev'])

                # Populate df
                df = pd.concat([df, tmp_df], axis=1)

                # Update index values
                df.index += 1

                # Save df as CSV file
                df.to_csv(os.path.join(save_path, save_dir + '_all_df.csv'))

                # Specify predictions file name and path
                pred_name = save_dir + '_train_pred.mat'
                pred_name = os.path.join(save_path, pred_name)

                # Save predictions as MAT file
                sio.savemat(pred_name, {'train_pred': np.transpose(train_pred.cpu().numpy(), (1, 2, 0))})

                # Specify predictions file name and path
                pred_name = save_dir + '_eval_pred.mat'
                pred_name = os.path.join(save_path, pred_name)

                # Save predictions as MAT file
                sio.savemat(pred_name, {'eval_pred': np.transpose(eval_pred.cpu().numpy(), (1, 2, 0))})

                # Specify segmentations file name and path
                seg_name = save_dir + '_train_seg.mat'
                seg_name = os.path.join(save_path, seg_name)

                # Save segmentations as MAT file
                sio.savemat(seg_name, {'train_seg': np.transpose(train_seg.cpu().numpy(), (1, 2, 0))})

                # Specify segmentations file name and path
                seg_name = save_dir + '_eval_seg.mat'
                seg_name = os.path.join(save_path, seg_name)

                # Save segmentations as MAT file
                sio.savemat(seg_name, {'eval_seg': np.transpose(eval_seg.cpu().numpy(), (1, 2, 0))})

                # Calculate Dice coefficient of each class in each transformed moving segmentation wrt predicted segmentations
                eval_dc_array = EachClassDC().loss(from_numpy(int64(tmp_seg_array[subj_array == val_subj, ...])), eval_seg.cpu()).numpy()

                # Create pandas DataFrame of Dice coefficients
                df = pd.DataFrame(eval_dc_array[:, 1:], columns=['head', 'soft_palate', 'jaw', 'tongue', 'vocal_tract', 'tooth_space'])

                # Update index values
                df.index += 1

                # Save df as CSV file
                df.to_csv(os.path.join(save_path, save_dir + '_df_dc_pred.csv'))

                # Calculate Dice coefficient of each class in each transformed moving segmentation wrt ground-truth segmentations
                eval_dc_array = EachClassDC().loss(from_numpy(int64(seg_array[subj_array == val_subj, ...])), eval_seg.cpu()).numpy()

                # Create pandas DataFrame of Dice coefficients
                df = pd.DataFrame(eval_dc_array[:, 1:], columns=['head', 'soft_palate', 'jaw', 'tongue', 'vocal_tract', 'tooth_space'])

                # Update index values
                df.index += 1

                # Save df as CSV file
                df.to_csv(os.path.join(save_path, save_dir + '_df_dc_gt.csv'))

                # Stop timing experiment
                exp_end = time.time()
                exp_min, exp_sec = divmod(exp_end - exp_start, 60)
                exp_hou, exp_min = divmod(exp_min, 60)
                print('Time elapsed in hh:mm:ss: {:02.0f}:{:02.0f}:{:02.0f}'.format(exp_hou, exp_min, exp_sec))

    else:

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

        # Create training dataset
        train_ds, aug_ds = train_ds_fn(test_subj, img_array, tmp_seg_array, seg_array, subj_array)

        # Create other training dataset and evaluating dataset
        other_train_ds, eval_ds = train_eval_ds_fn(test_subj, img_array, tmp_seg_array, seg_array, subj_array)

        # Create batches of training and evaluating data
        traindataloader = DataLoader(train_ds, batch_size = mb_size,
                            shuffle = True, num_workers = 4)
        alltraindataloader = DataLoader(other_train_ds, batch_size = len(other_train_ds),
                            shuffle = False, num_workers = 4)
        singletraindataloader = DataLoader(other_train_ds, batch_size = 1,
                            shuffle = False, num_workers = 4)                     
        allevaldataloader = DataLoader(eval_ds, batch_size = len(eval_ds),
                            shuffle = False, num_workers = 4)
        singleevaldataloader = DataLoader(eval_ds, batch_size = 1,
                            shuffle = False, num_workers = 4)

        # Extract learning rate, lambda and gamma
        l_rate = lr_array[fold]
        lambda_param = lambda_array[fold]
        gamma_param = gamma_array[fold]
            
        # Print summary
        print(f'Learning rate: {l_rate}   Lambda: {lambda_param}   Gamma: {gamma_param}')

        # Create U-Net model using VxmDense
        vxm_model = vxmwip.networks.VxmDense(inshape, nb_features, int_steps=0)

        # Zero gradient buffers of model
        vxm_model.zero_grad() 

        # Send model to GPU
        vxm_model.to(device)

        # Create the optimizer
        optimizer = torch.optim.Adam(vxm_model.parameters(), lr = l_rate)

        # Weighting for each loss
        loss_weights = [1, lambda_param, gamma_param]
        
        # Preallocate arrays for losses
        all_train_losses_array = np.empty([n_epochs, len(losses)])
        all_eval_losses_array = np.empty([n_epochs, len(losses)])
        single_train_losses_array = np.empty([len(singletraindataloader), n_epochs * len(losses)])
        single_eval_losses_array = np.empty([len(singleevaldataloader), n_epochs * len(losses)])

        # Preallocate arrays for Dice coefficients
        all_train_dc_array = np.empty(n_epochs)
        all_train_dc_pred_array = np.empty(n_epochs)
        all_eval_dc_array = np.empty(n_epochs)
        all_eval_dc_pred_array = np.empty(n_epochs)
        single_train_dc_array = np.empty([len(singletraindataloader), n_epochs])
        single_train_dc_pred_array = np.empty([len(singletraindataloader), n_epochs])
        single_eval_dc_array = np.empty([len(singleevaldataloader), n_epochs])
        single_eval_dc_pred_array = np.empty([len(singleevaldataloader), n_epochs])

        # Train and evaluate network

        # For each epoch
        for epoch in range(n_epochs):
            
            # Measure time
            start_time = time.time()

            # Train network
            train(vxm_model, device, traindataloader, optimizer, losses, loss_weights)

            # Evaluate network using training dataset
            train_losses, train_pred, train_dc, train_dc_pred, train_seg = evaluate(vxm_model, device, singletraindataloader, inshape, losses, loss_weights)

            # Populate single_train_losses_array
            single_train_losses_array[:, (len(losses) * epoch):(len(losses) * (epoch + 1))] = train_losses

            # Populate single_train_dc_array and single_train_dc_pred_array
            single_train_dc_array[:, epoch] = train_dc
            single_train_dc_pred_array[:, epoch] = train_dc_pred

            # Evaluate network using training dataset
            train_losses, train_pred, train_dc, train_dc_pred, train_seg = evaluate(vxm_model, device, alltraindataloader, inshape, losses, loss_weights)

            # Populate all_train_losses_array
            all_train_losses_array[epoch, :] = train_losses

            # Populate all_train_dc_array and all_train_dc_pred_array
            all_train_dc_array[epoch] = train_dc
            all_train_dc_pred_array[epoch] = train_dc_pred

            # Evaluate network using evaluating dataset
            eval_losses, eval_pred, eval_dc, eval_dc_pred, eval_seg = evaluate(vxm_model, device, singleevaldataloader, inshape, losses, loss_weights)

            # Populate single_eval_losses_array
            single_eval_losses_array[:, (len(losses) * epoch):(len(losses) * (epoch + 1))] = eval_losses

            # Populate single_eval_dc_array and single_eval_dc_pred_array
            single_eval_dc_array[:, epoch] = eval_dc
            single_eval_dc_pred_array[:, epoch] = eval_dc_pred

            # Evaluate network using evaluating dataset
            eval_losses, eval_pred, eval_dc, eval_dc_pred, eval_seg = evaluate(vxm_model, device, allevaldataloader, inshape, losses, loss_weights)

            # Populate all_eval_losses_array
            all_eval_losses_array[epoch, :] = eval_losses

            # Populate all_eval_dc_array and all_eval_dc_pred_array
            all_eval_dc_array[epoch] = eval_dc
            all_eval_dc_pred_array[epoch] = eval_dc_pred

            # Print epoch and loss information
            print('Epoch: %d  Time: %.2f sec\nTraining loss: %.5f, %.5f, %.5f   Evaluating loss: %.5f, %.5f, %.5f\nTraining MCDC: %.2f   Evaluating MCDC: %.2f' %
                    (epoch + 1, time.time() - start_time,
                    train_losses[:, 0], train_losses[:, 1], train_losses[:, 2],
                    eval_losses[:, 0], eval_losses[:, 1], eval_losses[:, 2],
                    train_dc, eval_dc))

        print('Finished Training')

        # Specify folder where all files will be saved
        save_dir = f'test_{test_subj}_lr_{l_rate}_mbs_{mb_size}_epochs_{n_epochs}_mse_1_l2_{lambda_param}_dc_{gamma_param}_aug_{aug_ds}'
        save_path = os.path.join(f'/home/mr19/Documents/registration/voxelmorph/mse_l2_dc_aug_{aug_ds}_experiment_{exp_n}', save_dir)

        # Check if folder exists and create if necessary
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Specify network name and path
        network_name = save_dir + '_parameters.pth'
        network_name = os.path.join(save_path, network_name)

        # Save network parameters
        torch.save(vxm_model.state_dict(), network_name)

        # Specify losses file name and path
        losses_name = save_dir + '_single_train_loss.csv'
        losses_name = os.path.join(save_path, losses_name)

        # Save losses array in a csv file
        np.savetxt(losses_name, single_train_losses_array, delimiter = ',')

        # Preallocate pandas DataFrame for losses and Dice coefficients
        df = pd.DataFrame()

        # Create pandas DataFrame of training losses
        tmp_df = pd.DataFrame(all_train_losses_array, columns=['mse_tr', 'gl_tr', 'adc_tr'])
        
        # Calculate total training loss
        tmp_df['total_tr'] = tmp_df.sum(axis=1)

        # Populate df
        df = pd.concat([df, tmp_df], axis=1)

        # Specify losses file name and path
        losses_name = save_dir + '_single_eval_loss.csv'
        losses_name = os.path.join(save_path, losses_name)

        # Save losses array in a csv file
        np.savetxt(losses_name, single_eval_losses_array, delimiter = ',')

        # Create pandas DataFrame of evaluating losses
        tmp_df = pd.DataFrame(all_eval_losses_array, columns=['mse_ev', 'gl_ev', 'adc_ev'])
        
        # Calculate total evaluating loss
        tmp_df['total_ev'] = tmp_df.sum(axis=1)

        # Populate df
        df = pd.concat([df, tmp_df], axis=1)

        # Specify Dice coefficient file name and path
        dc_name = save_dir + '_single_train_dc_gt.csv'
        dc_name = os.path.join(save_path, dc_name)

        # Save Dice coefficient array in a csv file
        np.savetxt(dc_name, single_train_dc_array, delimiter = ',')

        # Specify Dice coefficient file name and path
        dc_name = save_dir + '_single_train_dc_pred.csv'
        dc_name = os.path.join(save_path, dc_name)

        # Save Dice coefficient array in a csv file
        np.savetxt(dc_name, single_train_dc_pred_array, delimiter = ',')

        # Create pandas DataFrame of mean Dice coefficient
        tmp_df = pd.DataFrame(all_train_dc_array, columns=['dc_gt_tr'])

        # Populate df
        df = pd.concat([df, tmp_df], axis=1)

        # Create pandas DataFrame of mean Dice coefficient
        tmp_df = pd.DataFrame(all_train_dc_pred_array, columns=['dc_pred_tr'])

        # Populate df
        df = pd.concat([df, tmp_df], axis=1)

        # Specify Dice coefficient file name and path
        dc_name = save_dir + '_single_eval_dc_gt.csv'
        dc_name = os.path.join(save_path, dc_name)

        # Save Dice coefficient array in a csv file
        np.savetxt(dc_name, single_eval_dc_array, delimiter = ',')

        # Specify Dice coefficient file name and path
        dc_name = save_dir + '_single_eval_dc_pred.csv'
        dc_name = os.path.join(save_path, dc_name)

        # Save Dice coefficient array in a csv file
        np.savetxt(dc_name, single_eval_dc_pred_array, delimiter = ',')

        # Create pandas DataFrame of mean Dice coefficient
        tmp_df = pd.DataFrame(all_eval_dc_array, columns=['dc_gt_ev'])

        # Populate df
        df = pd.concat([df, tmp_df], axis=1)

        # Create pandas DataFrame of mean Dice coefficient
        tmp_df = pd.DataFrame(all_eval_dc_pred_array, columns=['dc_pred_ev'])

        # Populate df
        df = pd.concat([df, tmp_df], axis=1)

        # Update index values
        df.index += 1

        # Save df as CSV file
        df.to_csv(os.path.join(save_path, save_dir + '_all_df.csv'))

        # Specify predictions file name and path
        pred_name = save_dir + '_train_pred.mat'
        pred_name = os.path.join(save_path, pred_name)

        # Save predictions as MAT file
        sio.savemat(pred_name, {'train_pred': np.transpose(train_pred.cpu().numpy(), (1, 2, 0))})

        # Specify predictions file name and path
        pred_name = save_dir + '_eval_pred.mat'
        pred_name = os.path.join(save_path, pred_name)

        # Save predictions as MAT file
        sio.savemat(pred_name, {'eval_pred': np.transpose(eval_pred.cpu().numpy(), (1, 2, 0))})

        # Specify segmentations file name and path
        seg_name = save_dir + '_train_seg.mat'
        seg_name = os.path.join(save_path, seg_name)

        # Save segmentations as MAT file
        sio.savemat(seg_name, {'train_seg': np.transpose(train_seg.cpu().numpy(), (1, 2, 0))})

        # Specify segmentations file name and path
        seg_name = save_dir + '_eval_seg.mat'
        seg_name = os.path.join(save_path, seg_name)

        # Save segmentations as MAT file
        sio.savemat(seg_name, {'eval_seg': np.transpose(eval_seg.cpu().numpy(), (1, 2, 0))})

        # Calculate Dice coefficient of each class in each transformed moving segmentation wrt predicted segmentations
        eval_dc_array = EachClassDC().loss(from_numpy(int64(tmp_seg_array[subj_array == test_subj, ...])), eval_seg.cpu()).numpy()

        # Create pandas DataFrame of Dice coefficients
        df = pd.DataFrame(eval_dc_array[:, 1:], columns=['head', 'soft_palate', 'jaw', 'tongue', 'vocal_tract', 'tooth_space'])

        # Update index values
        df.index += 1

        # Save df as CSV file
        df.to_csv(os.path.join(save_path, save_dir + '_df_dc_pred.csv'))

        # Calculate Dice coefficient of each class in each transformed moving segmentation wrt ground-truth segmentations
        eval_dc_array = EachClassDC().loss(from_numpy(int64(seg_array[subj_array == test_subj, ...])), eval_seg.cpu()).numpy()

        # Create pandas DataFrame of Dice coefficients
        df = pd.DataFrame(eval_dc_array[:, 1:], columns=['head', 'soft_palate', 'jaw', 'tongue', 'vocal_tract', 'tooth_space'])

        # Update index values
        df.index += 1

        # Save df as CSV file
        df.to_csv(os.path.join(save_path, save_dir + '_df_dc_gt.csv'))

        # Stop timing experiment
        exp_end = time.time()
        exp_min, exp_sec = divmod(exp_end - exp_start, 60)
        exp_hou, exp_min = divmod(exp_min, 60)
        print('Time elapsed in hh:mm:ss: {:02.0f}:{:02.0f}:{:02.0f}'.format(exp_hou, exp_min, exp_sec))