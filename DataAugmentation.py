# DataAugmentation.py
# Code to perform data augmentation

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 7th September 2023

# Import required modules
from numpy import float32, int64, newaxis, float64, zeros_like
from torch import from_numpy
from numpy.random import uniform, randint
from skimage.transform import rotate, resize
from math import floor, ceil, sin, radians


# Create a transform
class ToTensor(object):
    
    # To convert NumPy arrays in a sample to PyTorch tensors
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