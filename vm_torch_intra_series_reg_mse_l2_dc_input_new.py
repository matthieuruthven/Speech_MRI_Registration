# vm_torch_intra_series_reg_mse_l2_dc_input_new.py
# A script to perform intra series image registration using VoxelMorph
# with a PyTorch backend

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 31st December 2023

# Import required module
import argparse
from pathlib import Path
import torch
import os
import numpy as np
from scipy.io import loadmat, savemat
from torchvision import transforms
from DataAugmentation import ToTensor, RotateCropAndPad, RandomCrop, Rescale, RandomTranslation, RescaleAndPad
from SpeechMRIDatasets import SpeechMRIData, SpeechMRIDataRandom
from torch.utils.data import ConcatDataset, DataLoader
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch' # Import VoxelMorph with PyTorch backend
import voxelmorph as vxm
# import voxelmorphwip as vxmwip
import torch.nn.functional as F
# from monai.metrics import DiceMetric
# import pandas as pd


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
    frame = loadmat(full_img_list[0])['image_frame']

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
        img_array[idx, ...] = loadmat(file_path)['image_frame']

    # For each GT segmentation
    for idx, file_path in enumerate(full_gt_list):

        # Populate gt_array
        gt_array[idx, ...] = loadmat(file_path)['mask_frame'].astype('int64')

    for idx, file_path in enumerate(full_seg_list):

        # Populate seg_array
        seg_array[idx, ...] = loadmat(file_path)['mask_frame'].astype('int64')

    # If required, augment the dataset
    if augmentation and random_choice:

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

    else:

        # Define transformation
        data_transform = transforms.Compose([ToTensor()])

        # Create dataset
        dataset = SpeechMRIData(img_array, seg_array, gt_array, subj_id_array, transform = data_transform)

    return dataset, first_frame_dim


def main(data_dir, train_subj, val_subj, epochs, l_rate, alpha_param, gamma_param):

    """Function performs the following steps:
       1) Loads images for training (and validation)
       2) Sets up the VoxelMorph model
       3) Trains (and validates) the model
       4) Saves model parameters and training (and validation) losses"""

    # Create training dataset
    training_dataset, inshape = create_dataset(data_dir, train_subj)

    # Print update on training dataset
    print(f'Training dataset created consisting of images of subject(s) {train_subj}')

    # If required, create a validation dataset
    if val_subj:
        validation_dataset, inshape = create_dataset(data_dir, val_subj, augmentation=False)
        
        # Print update on validation dataset
        print(f'Validation dataset created consisting of images of subject(s) {val_subj}')
    else:
        # Print update on validation dataset
        print(f'No validation dataset')

    # Create dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers=4)
    if val_subj:
        validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=4)

    # Specify GPU where training will occur
    device = torch.device("cuda:0")

    # U-Net model architecture
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    nb_features = [enc_nf, dec_nf]

    # Create U-Net model using VxmDense
    vxm_model = vxmwip.networks.VxmDense(inshape, nb_features, int_steps=0)

    # Send model to GPU
    vxm_model.to(device)

    # Zero gradient buffers of model
    vxm_model.zero_grad() 

    # Losses in loss function
    losses = [vxm.losses.MSE().loss, vxmwip.losses.Grad('l2').loss, vxm.losses.Dice().loss]

    # Weighting for each loss
    loss_weights = [1, lambda_param, gamma_param]
                
    # Create the optimizer
    optimizer = torch.optim.Adam(vxm_model.parameters(), lr = l_rate)

    # Function to calculate Dice coefficients of estimated segmentations
    calc_dsc = DiceMetric(include_background=False, reduction='none')

    # Preallocate lists for training (and validation) losses (and validation Dice coefficients)
    train_loss_list = []
    if val_subj:
        val_loss_list = []
        mean_dsc_list = []
        mean_dsc_per_class = torch.zeros((6, epochs), dtype=torch.float, device=device)

    # Print updates
    print('Training of segmentation CNN started')
    print(f'Training duration: {epochs} epochs')
    print(f'Learning rate: {l_rate}')
    print(f'Alpha value: {alpha_param}')
    print(f'Gamma value: {gamma_param}')

    # For each epoch
    for epoch in range(1, epochs + 1):

        # Training loss
        training_loss = 0.

        # Set model to training mode
        vxm_model.train()

        # For each mini-batch in training dataset
        for speech_data in training_dataloader:
            
            # Extract images and segmentations and then send them to GPU
            mov_img = speech_data['mov_img'].to(device)
            mov_seg = speech_data['mov_seg'].to(device)
            mov_gt = speech_data['mov_gt'].to(device)
            fix_img = speech_data['fix_img'].to(device)
            fix_seg = speech_data['fix_seg'].to(device)
            fix_gt = speech_data['fix_gt'].to(device)

            # Create tensor of zeros (required for deformation field)
            zero_phi = torch.cat((torch.zeros_like(mov_img), torch.zeros_like(mov_img)), 1).to(device)

            # Create required lists
            input = [torch.cat((mov_img, mov_seg[:, 1:, ...]), 1), torch.cat((fix_img, fix_seg[:, 1:, ...]), 1), mov_gt[:, 1:, ...]]
            y_true = [torch.cat((fix_img, fix_gt[:, 1:, ...]), 1), zero_phi]

            # Zero the optimiser gradients
            optimizer.zero_grad()

            # Forward propagation
            y_pred = vxm_model(*input)
            
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

            # Update training loss
            training_loss += loss.item()

        # Calculate mean training loss and update train_loss_list
        tmp_training_loss = training_loss / len(training_dataloader)
        train_loss_list.append(tmp_training_loss)

        # If required, calculate validation loss
        if val_subj:
            
            # Set model to evaluation mode
            vxm_model.eval()
            with torch.no_grad():

                # For each image in validation dataset
                for speech_data in validation_dataloader:

                    # Extract images and segmentations and then send them to GPU
                    mov_img = speech_data['mov_img'].to(device)
                    mov_seg = speech_data['mov_seg'].to(device)
                    mov_gt = speech_data['mov_gt'].to(device)
                    fix_img = speech_data['fix_img'].to(device)
                    fix_seg = speech_data['fix_seg'].to(device)
                    fix_gt = speech_data['fix_gt'].to(device)

                    # Create tensor of zeros (required for deformation field)
                    zero_phi = torch.cat((torch.zeros_like(mov_img), torch.zeros_like(mov_img)), 1).to(device)

                    # Create required lists
                    input = [torch.cat((mov_img, mov_seg[:, 1:, ...]), 1), torch.cat((fix_img, fix_seg[:, 1:, ...]), 1), mov_gt[:, 1:, ...]]
                    y_true = [torch.cat((fix_img, fix_gt[:, 1:, ...]), 1), zero_phi]

                    # Forward propagation
                    y_pred = vxm_model(*input)
                    
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

                    # Update val_loss_list
                    val_loss_list.append(loss.item())

                    # Create spatial transformer
                    s_transformer = vxm.layers.SpatialTransformer(inshape, mode='nearest')

                    # Move spatial transformer to GPU
                    s_transformer.to(device)

                    # Transform segmentations
                    tra_mov_gt = s_transformer(mov_gt, y_pred[1])

                    # Estimated segmentations
                    est_segs = torch.argmax(tra_mov_gt, dim = 1)

                    # One-hot encode segmentations
                    tmp_est_segs = F.one_hot(est_segs, num_classes=n_classes)
                    labels = F.one_hot(labels, num_classes=n_classes)

                    # Permute dimensions of segmentations
                    tmp_est_segs = torch.permute(tmp_est_segs, (0, 3, 1, 2))
                    labels = torch.permute(labels, (0, 3, 1, 2))

                    # Calculate the Dice coefficient
                    dsc = calc_dsc(tmp_est_segs, labels)

                    # Update mean_dsc_per_class
                    mean_dsc_per_class[:, epoch - 1] = torch.mean(dsc, dim=0)

                    # Update mean_dsc_list
                    dsc = torch.mean(dsc).item()
                    mean_dsc_list.append(dsc)
    
        # Print epoch and loss information
        if val_subj:
            print(f'Epoch: {epoch}   Training loss: {tmp_training_loss:.4f}   Validation loss: {validation_loss.item():.4f}   Dice coefficient: {dsc:.4f}')
        else:
            print(f'Epoch: {epoch}   Training loss: {tmp_training_loss:.4f}')

    # Print update
    print('Finished segmentation CNN training')

    # Create string of training dataset subject IDs
    train_subj_string = [str(f) for f in train_subj]
    train_subj_string = "_".join(train_subj_string)

    # Path to folder where parameters of trained model will be saved
    if val_subj:

        # Create string of training dataset subject IDs
        val_subj_string = [str(f) for f in val_subj]
        val_subj_string = "_".join(val_subj_string)

        # Path to folder
        save_dir_path = data_dir / 'RegistrationCNNs' / f'val_subj_{val_subj_string}_train_subj_{train_subj_string}_l_rate_{l_rate}_mb_size_{mb_size}_epochs_{epochs}'    
    else:
        save_dir_path = data_dir / 'RegistrationCNNs' / f'train_subj_{"_".join(train_subj)}_l_rate_{l_rate}_mb_size_{mb_size}_epochs_{epochs}'
    
    # If required, create folders
    if os.path.exists(save_dir_path):
        # Print update
        print(f'{save_dir_path} already exists')
    else:
        os.makedirs(save_dir_path)
        # Print update
        print(f'{save_dir_path} created')
    
    # Save parameters of trained model
    torch.save(vxm_model.state_dict(), save_dir_path / 'vxm_parameters.pth')

    # Print update
    print(f'Parameters of trained registration CNN saved here: "{save_dir_path / "vxm_parameters.pth"}"')

    # Create a pandas DataFrame of training (and validation) losses
    df = pd.DataFrame({'Frame': range(1, epochs + 1),
                       'MeanLoss': train_loss_list,
                       'LossType': 'Training'})
    if val_subj:
        tmp_df = pd.DataFrame({'Frame': range(1, epochs + 1),
                               'MeanLoss': val_loss_list,
                               'LossType': 'Validation'})
        df = pd.concat([df, tmp_df])
    
    # Save df
    df.to_csv(save_dir_path / 'mean_losses.csv', index=False)
    
    # Print update
    print(f'Losses saved here: "{save_dir_path / "mean_losses.csv"}"')

    # If required, save Dice coefficients of transformed segmentations (validation dataset)
    if val_subj:
        
        # Create a pandas DataFrame of Dice coefficients
        df = pd.DataFrame({'Frame': range(1, epochs + 1),
                           'MeanDSC': mean_dsc_list,
                           'Class': 'Overall'})
        
        # Convert mean_dsc_per_class from PyTorch tensor to NumPy array
        mean_dsc_per_class = mean_dsc_per_class.cpu().numpy()

        # Create a pandas DataFrame of Dice coefficients
        for idx, class_name in enumerate(['Head', 'SoftPalate', 'Jaw', 'Tongue', 'VocalTract', 'ToothSpace']):
            tmp_df = pd.DataFrame({'Frame': range(1, epochs + 1),
                                   'MeanDSC': mean_dsc_per_class[idx, :],
                                   'Class': class_name})
            df = pd.concat([df, tmp_df])
        
        # Save df
        df.to_csv(save_dir_path / 'mean_dsc.csv', index=False)

        # Print update
        print(f'Mean Dice coefficients saved here: "{save_dir_path / "mean_dsc.csv"}"')

        # Save transformed segmentations
        savemat(save_dir_path / 'transformed_segmentations.mat', {'tra_segs': np.uint8(torch.permute(est_segs, (1, 2, 0)).cpu().numpy())})

        # Print update
        print(f'Segmentations transformed according to displacement fields estimated by registration CNN saved here: "{save_dir_path / "transformed_segmentations.mat"}"')


if __name__ == "__main__":
  
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Code to train a convolutional neural network to segment images.')
    
    # Add arguments
    parser.add_argument(
        '--data_dir', 
        help='The path to the folder containing all the data.',
        default='data',
        type=Path
        )
    parser.add_argument(
        '--train_subj',
        help='A list of ID(s) of subjects to use in the training dataset.',
        type=int,
        default=1,
        nargs='*'
        )
    parser.add_argument(
        '--val_subj',
        help='A list of ID(s) of subjects to use in the validation dataset.',
        type=int,
        default=2,
        nargs='*'
        )
    parser.add_argument(
        '--epochs',
        help='Number of epochs of training.',
        type=int,
        default=200
        )
    parser.add_argument(
        '--l_rate',
        help='Learning rate to use in training.',
        type=float,
        default=0.0009
        )
    parser.add_argument(
        '--alpha',
        help='Weighting of L2 regularisation term in loss function.',
        type=float,
        default=0.01
        )
    parser.add_argument(
        '--gamma',
        help='Weighting of Dice coefficient term in loss function.',
        type=float,
        default=1.0
        )
    parser.add_argument(
        '--gpu_id',
        help='ID of GPU to use for training.',
        type=int,
        default=0
        )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if CUDA is available
    # assert torch.cuda.is_available(), 'PyTorch does not detect any GPUs.'

    # Select GPU to use for training
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Check if data_dir exists
    assert os.path.exists(args.data_dir), 'Please specify the absolute path to the folder containing all the data using the --data_dir argument to "TrainCNN.py".'

    # Check if images have been normalised
    assert os.path.exists(args.data_dir / 'Normalised_Images'), 'Have the images been normalised? This can be done using "NormaliseImages.py".'
    
    # Check that args.train_subj is not empty
    assert args.train_subj, f'Please specify the IDs of the subjects whose datasets should be included in the training dataset using the --train_subj argument to "TrainCNN.py".'
    
    # If required, modify args.train_subj
    if isinstance(args.train_subj, int):
        args.train_subj = [args.train_subj]

    # If required, modify args.val_subj
    if isinstance(args.val_subj, int):
        args.val_subj = [args.val_subj]

    # Run main function
    main(args.data_dir, args.train_subj, args.val_subj, args.epochs, args.l_rate, args.alpha, args.gamma)