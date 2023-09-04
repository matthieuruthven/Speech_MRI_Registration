# vm_torch_intra_series_reg_mse_l2_dc_input_new.py
# A script to perform intra series image registration using VoxelMorph
# with a PyTorch backend

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 15th February 2022

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
class SpeechMRIDataRandom(Dataset):
    
    # Dataset of two MR images (moving and fixed image) and the corresponding predicted
    # and ground-truth segmentations of these images
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
    
        # Extract another image of subject and corresponding predicted and ground-truth segmentations at random
        jdx = randint(subj_img.shape[0])
        mov_img = subj_img[jdx, ...]
        mov_seg = subj_seg[jdx, ...]
        mov_gt = subj_gt[jdx, ...]

        # Define sample
        sample = {'mov_img': mov_img, 'mov_seg': mov_seg, 'mov_gt': mov_gt, 'fix_img': fix_img, 'fix_seg': fix_seg, 'fix_gt': fix_gt}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

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

# Create a transform
class RandomCrop(object):
    
    # Random crop with top left corner within x_range and y_range
    # Args:
    # 1) x_range (tuple) the range of possible x-coordinates for the top left corner of the crop
    # 2) y_range (tuple) the range of possible y-coordinates for the top left corner of the crop
    # 3) output_size (int or tuple) the dimensions of the crop. If int, the crop is square

    def __init__(self, x_range, y_range, output_size):
        self.x_range = x_range
        self.y_range = y_range
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        mov_img, mov_seg, mov_gt, fix_img, fix_seg, fix_gt = sample['mov_img'], sample['mov_seg'], sample['mov_gt'], sample['fix_img'], sample['fix_seg'], sample['fix_gt']

        new_h, new_w = self.output_size

        top = randint(self.y_range[0], self.y_range[1])
        left = randint(self.x_range[0], self.x_range[1])

        mov_img = mov_img[top: top + new_h,
                      left: left + new_w]

        mov_seg = mov_seg[top: top + new_h,
                      left: left + new_w]

        mov_gt = mov_gt[top: top + new_h,
                      left: left + new_w]

        fix_img = fix_img[top: top + new_h,
                      left: left + new_w]

        fix_seg = fix_seg[top: top + new_h,
                      left: left + new_w]
        
        fix_gt = fix_gt[top: top + new_h,
                      left: left + new_w]

        return {'mov_img': mov_img, 'mov_seg': mov_seg, 'mov_gt': mov_gt, 'fix_img': fix_img, 'fix_seg': fix_seg, 'fix_gt': fix_gt}

# Import required modules
from numpy import zeros_like

# Create a transform
class RandomTranslation(object):
    
    # Random translation with top left corner within x_range and y_range
    # Args:
    # 1) x_range (tuple) the range of possible x-coordinates for the top left corner of the translated image
    # 2) y_range (tuple) the range of possible y-coordinates for the top left corner of the translated image

    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

    def __call__(self, sample):
        mov_img, mov_seg, mov_gt, fix_img, fix_seg, fix_gt = sample['mov_img'], sample['mov_seg'], sample['mov_gt'], sample['fix_img'], sample['fix_seg'], sample['fix_gt']

        # Find size of images and segmentations
        h, w = mov_img.shape

        # Randomly choose translation
        top = randint(self.y_range[0], self.y_range[1])
        left = randint(self.x_range[0], self.x_range[1])

        # Define translated images and segmentations
        translated_mov_img = zeros_like(mov_img)
        translated_mov_seg = zeros_like(mov_seg)
        translated_mov_gt = zeros_like(mov_gt)
        translated_fix_img = zeros_like(fix_img)
        translated_fix_seg = zeros_like(fix_seg)
        translated_fix_gt = zeros_like(fix_gt)

        # Populate translated images and segmentations
        if left > 0:
            translated_mov_img[top:, left:] = mov_img[:h - top, :w - left]
            translated_mov_seg[top:, left:] = mov_seg[:h - top, :w - left]
            translated_mov_gt[top:, left:] = mov_gt[:h - top, :w - left]
            translated_fix_img[top:, left:] = fix_img[:h - top, :w - left]
            translated_fix_seg[top:, left:] = fix_seg[:h - top, :w - left]
            translated_fix_gt[top:, left:] = fix_gt[:h - top, :w - left]
        else:
            translated_mov_img[top:, :w + left] = mov_img[:h - top:, -left:]
            translated_mov_seg[top:, :w + left] = mov_seg[:h - top, -left:]
            translated_mov_gt[top:, :w + left] = mov_gt[:h - top, -left:]
            translated_fix_img[top:, :w + left] = fix_img[:h - top:, -left:]
            translated_fix_seg[top:, :w + left] = fix_seg[:h - top, -left:]
            translated_fix_gt[top:, :w + left] = fix_gt[:h - top, -left:]

        return {'mov_img': translated_mov_img,
                'mov_seg': translated_mov_seg,
                'mov_gt': translated_mov_gt,
                'fix_img': translated_fix_img,
                'fix_seg': translated_fix_seg,
                'fix_gt': translated_fix_gt}

# Import required modules
from numpy.random import uniform
from numpy import float64
from skimage.transform import rotate
from math import floor, ceil, cos, sin, radians

# Create a transform
class RotateCropAndPad(object):
    
    # Rotate, crop and then zero pad the images and segmentations in a sample so that 
    # their dimensions do not change
    # Arg:
    # 1) rotation_range (tuple) the range of angles in degrees that the image can be rotated by

    def __init__(self, rotation_range):
        self.rotation_range = rotation_range

    def __call__(self, sample):
        mov_img, mov_seg, mov_gt, fix_img, fix_seg, fix_gt = sample['mov_img'], sample['mov_seg'], sample['mov_gt'], sample['fix_img'], sample['fix_seg'], sample['fix_gt']

        # Find size of images and segmentations
        h, w = mov_img.shape

        # Randomly choose rotation
        rotation_angle = uniform(self.rotation_range[0], self.rotation_range[1])

        # Convert segmentations to correct data type
        mov_seg = float64(mov_seg)
        mov_gt = float64(mov_gt)
        fix_seg = float64(fix_seg)
        fix_gt = float64(fix_gt)

        # Rotate images and segmentations
        mov_img = rotate(mov_img, rotation_angle, order = 0) # 0: nearest neighbour
        mov_seg = rotate(mov_seg, rotation_angle, order = 0) # 0: nearest neighbour
        mov_gt = rotate(mov_gt, rotation_angle, order = 0) # 0: nearest neighbour
        fix_img = rotate(fix_img, rotation_angle, order = 0) # 0: nearest neighbour
        fix_seg = rotate(fix_seg, rotation_angle, order = 0) # 0: nearest neighbour
        fix_gt = rotate(fix_gt, rotation_angle, order = 0) # 0: nearest neighbour

        # Define rotated, cropped and padded images and segmentations
        rotated_mov_img = zeros_like(mov_img)
        rotated_mov_seg = zeros_like(mov_seg)
        rotated_mov_gt = zeros_like(mov_gt)
        rotated_fix_img = zeros_like(fix_img)
        rotated_fix_seg = zeros_like(fix_seg)
        rotated_fix_gt = zeros_like(fix_gt)

        # Convert rotation_angle to radians
        rotation_angle = radians(rotation_angle)

        # Crop and pad rotated image frame and mask frame
        if rotation_angle > 0:
            rotated_mov_img[ceil(w / 2 * sin(rotation_angle)):, :] = mov_img[:floor(h - w / 2 * sin(rotation_angle)), :]
            rotated_mov_seg[ceil(w / 2 * sin(rotation_angle)):, :] = mov_seg[:floor(h - w / 2 * sin(rotation_angle)), :]
            rotated_mov_gt[ceil(w / 2 * sin(rotation_angle)):, :] = mov_gt[:floor(h - w / 2 * sin(rotation_angle)), :]
            rotated_fix_img[ceil(w / 2 * sin(rotation_angle)):, :] = fix_img[:floor(h - w / 2 * sin(rotation_angle)), :]
            rotated_fix_seg[ceil(w / 2 * sin(rotation_angle)):, :] = fix_seg[:floor(h - w / 2 * sin(rotation_angle)), :]
            rotated_fix_gt[ceil(w / 2 * sin(rotation_angle)):, :] = fix_gt[:floor(h - w / 2 * sin(rotation_angle)), :]
        else:
            rotated_mov_img[ceil(w / 4 * -sin(rotation_angle)):, :] = mov_img[:floor(h + w / 4 * sin(rotation_angle)), :]
            rotated_mov_seg[ceil(w / 4 * -sin(rotation_angle)):, :] = mov_seg[:floor(h + w / 4 * sin(rotation_angle)), :]
            rotated_mov_gt[ceil(w / 4 * -sin(rotation_angle)):, :] = mov_gt[:floor(h + w / 4 * sin(rotation_angle)), :]
            rotated_fix_img[ceil(w / 4 * -sin(rotation_angle)):, :] = fix_img[:floor(h + w / 4 * sin(rotation_angle)), :]
            rotated_fix_seg[ceil(w / 4 * -sin(rotation_angle)):, :] = fix_seg[:floor(h + w / 4 * sin(rotation_angle)), :]
            rotated_fix_gt[ceil(w / 4 * -sin(rotation_angle)):, :] = fix_gt[:floor(h + w / 4 * sin(rotation_angle)), :]

        return {'mov_img': rotated_mov_img,
                'mov_seg': rotated_mov_seg,
                'mov_gt': rotated_mov_gt,
                'fix_img': rotated_fix_img,
                'fix_seg': rotated_fix_seg,
                'fix_gt': rotated_fix_gt}

# Import required module
from skimage.transform import resize

# Create a transform
class Rescale(object):
    
    # Rescale the images and segmentations in a sample to a given size
    # Arg:
    # 1) output_size (int or tuple) the desired output size. If int, the size of the smaller edge
    # is matched to output_size while keeping the aspect ratio the same

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        mov_img, mov_seg, mov_gt, fix_img, fix_seg, fix_gt = sample['mov_img'], sample['mov_seg'], sample['mov_gt'], sample['fix_img'], sample['fix_seg'], sample['fix_gt']

        # Find size of images and segmentations
        h, w = mov_img.shape
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        # Convert segmentations to correct data type
        mov_seg = float64(mov_seg)
        mov_gt = float64(mov_gt)
        fix_seg = float64(fix_seg)
        fix_gt = float64(fix_gt)

        # Resize images and segmentations
        mov_img = resize(mov_img, (new_h, new_w), order = 0) # 0: nearest neighbour
        mov_seg = resize(mov_seg, (new_h, new_w), order = 0) # 0: nearest neighbour
        mov_gt = resize(mov_gt, (new_h, new_w), order = 0) # 0: nearest neighbour
        fix_img = resize(fix_img, (new_h, new_w), order = 0) # 0: nearest neighbour
        fix_seg = resize(fix_seg, (new_h, new_w), order = 0) # 0: nearest neighbour
        fix_gt = resize(fix_gt, (new_h, new_w), order = 0) # 0: nearest neighbour

        return {'mov_img': mov_img, 'mov_seg': mov_seg, 'mov_gt': mov_gt, 'fix_img': fix_img, 'fix_seg': fix_seg, 'fix_gt': fix_gt}

# Create a transform
class RescaleAndPad(object):
    
    # Rescale and then zero pad the images and segmentations in a sample so that their 
    # dimensions do not change
    # Arg:
    # 1) resize_range (tuple) the range of acceptable matrix dimensions following the rescale

    def __init__(self, resize_range):
        self.resize_range = resize_range

    def __call__(self, sample):
        mov_img, mov_seg, mov_gt, fix_img, fix_seg, fix_gt = sample['mov_img'], sample['mov_seg'], sample['mov_gt'], sample['fix_img'], sample['fix_seg'], sample['fix_gt']

        # Find size of images and segmentations
        h, w = mov_img.shape

        # Randomly choose new size of image from within resize_range
        new_size = randint(self.resize_range[0], self.resize_range[1])

        # Randomly choose lateral translation
        dx = randint(0, w - new_size)

        # Convert segmentations to correct data type
        mov_seg = float64(mov_seg)
        mov_gt = float64(mov_gt)
        fix_seg = float64(fix_seg)
        fix_gt = float64(fix_gt)

        # Preallocate arrays for rescaled and zero padded images and segmentations
        new_mov_img = zeros_like(mov_img)
        new_mov_seg = zeros_like(mov_seg)
        new_mov_gt = zeros_like(mov_gt)
        new_fix_img = zeros_like(fix_img)
        new_fix_seg = zeros_like(fix_seg)
        new_fix_gt = zeros_like(fix_gt)

        # Resize images and segmentations
        mov_img = resize(mov_img, (new_size, new_size), order = 0) # 0: nearest neighbour
        mov_seg = resize(mov_seg, (new_size, new_size), order = 0) # 0: nearest neighbour
        mov_gt = resize(mov_gt, (new_size, new_size), order = 0) # 0: nearest neighbour
        fix_img = resize(fix_img, (new_size, new_size), order = 0) # 0: nearest neighbour
        fix_seg = resize(fix_seg, (new_size, new_size), order = 0) # 0: nearest neighbour
        fix_gt = resize(fix_gt, (new_size, new_size), order = 0) # 0: nearest neighbour

        # Zero pad resized images and segmentations
        new_mov_img[h - new_size:, dx:dx + new_size] = mov_img
        new_mov_seg[h - new_size:, dx:dx + new_size] = mov_seg
        new_mov_gt[h - new_size:, dx:dx + new_size] = mov_gt
        new_fix_img[h - new_size:, dx:dx + new_size] = fix_img
        new_fix_seg[h - new_size:, dx:dx + new_size] = fix_seg
        new_fix_gt[h - new_size:, dx:dx + new_size] = fix_gt

        return {'mov_img': new_mov_img,
                'mov_seg': new_mov_seg,
                'mov_gt': new_mov_gt,
                'fix_img': new_fix_img,
                'fix_seg': new_fix_seg,
                'fix_gt': new_fix_gt}

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

# Import required modules
from torchvision import transforms
from torch.utils.data import ConcatDataset

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