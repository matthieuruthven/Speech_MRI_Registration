# SpeechMRIDatasets.py
# Code to create custom datasets of two-dimensional real-time 
# speech magnetic resonance (MR) images of the vocal tract during
# speech, and corresponding ground-truth (GT) segmentations

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 3rd March 2023

# Import required modules
from torch import is_tensor
from torch.utils.data import Dataset
from scipy.io import loadmat
from numpy.random import randint


# Create a custom dataset
class SpeechMRIDataRandom(Dataset):
    
    # Dataset of two MR images (moving and fixed image), and the estimated 
    # and GT segmentations of these images
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
    
    # Dataset of two MR images (moving and fixed image), and the estimated 
    # and GT segmentations of these images
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