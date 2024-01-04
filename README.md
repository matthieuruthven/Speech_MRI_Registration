# Speech_MRI_Registration
Code to perform nonlinear registration of magnetic resonance (MR) images of speech using the method developed by [Ruthven, Miquel and King (2023)](https://www.sciencedirect.com/science/article/pii/S1746809422007443). The method is deep learning based and inspired by [VoxelMorph](https://github.com/voxelmorph/voxelmorph). It includes a convolutional neural network (CNN) to estimate displacement fields to register pairs of images. The code to implement the method has been designed to be compatible with publicly available speech MR imaging (MRI) datasets ([available here](https://zenodo.org/records/10046815)). The datasets are described in detail in [Ruthven et al. (2023)](https://www.nature.com/articles/s41597-023-02766-z). The code should be compatible with other datasets if they are structured in the same way as the publicly available speech MRI datasets and follow the same naming convention.

## Introduction

### Requirements

1. Software: [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Hardware: Computer with one or more graphical processing units (GPUs) with memory > GB

### Setting up

1. Download this repository and the publicly available speech MRI datasets ([available here](https://zenodo.org/records/10046815)).

2. Open a new terminal and navigate to the folder containing the files from this repository.

3. Enter the following command to create a conda environment and install the Python packages required to run the code:
```
conda env create -f environment.yml
```

4. Enter the following command to activate the conda environment
```
conda activate SpeechMRIReg
```

5. Add the *data* folder containing the speech MRI datasets to the folder containing the files from this repository:
```
.
├── data
│   ├── GT_Segmentations
│   ├── MRI_SSFP_10fps
│   └── Velopharyngeal_Closure
├── CheckData.py
├── DataAugmentation.py
├── environment.yml
├── EvaluateCNN.py
├── NormaliseImages.py
├── README.md
├── SpeechMRIDatasets.py
└── TrainCNN.py
```

6. The method requires either estimated or ground-truth (GT) segmentations as an input. The code assumes that these segmentations are:
    - Contained in a folder called *Est_Segmentations* in the *data* folder;
    - Structured in the same way as those in the *GT_Segmentations* folder and follow the same naming convention.

    If using GT segmentations, simply copy the *GT_Segmentations* folder and rename this to *Est_Segmentations*. If using estimated segmentations, create a folder called *Est_Segmentations* in the *data* folder and then add the estimated segmentations to this folder. The *data* folder should then contain the following folders:  
```
.
├── data
│   ├── Est_Segmentations
│   ├── GT_Segmentations
│   ├── MRI_SSFP_10fps
│   └── Velopharyngeal_Closure
.
.
.
└── TrainCNN.py
```

7. Enter the following command to check that the speech MRI datasets are correctly structured and are not corrupted:
```
python CheckData.py
```
By default, CheckData.py assumes that the folder containing the datasets is called *data*, and that the data of subject 1 should be checked. However, these defaults can be overridden using the following arguments:
```
python CheckData.py --data_dir /path/to/folder --subj_id_list 1 2 4
```
In the example above, only the data of subjects 1, 2 and 4 would be checked.

8. Enter the following command to normalise the images of the datasets:
```
python NormaliseImages.py
```
By default, NormaliseImages.py assumes that the folder containing the dataset is called *data*, and that the data of subject 1 should be normalised. However, these defaults can be overridden using the following arguments:
```
python NormaliseImages.py --data_dir /path/to/folder --subj_id_list 3 5
```
In the example above, only the images of subjects 3 and 5 would be normalised.

## Training the registration method from scratch

Enter the following command to train the registration method from scratch:
```
python TrainCNN.py
```
By default, TrainCNN.py makes the following assumptions:
- Name of folder containing the entire dataset: *data*
- Subject to include in training dataset: 1
- Subject to include in validation dataset: 2
- Number of epochs of training: 200
- Learning rate to use in training: 0.0009
- Lambda parameter value: 0.01
- Gamma parameter value: 1.0
- ID of GPU to use in training: 0

However, these assumptions can be overridden using the following arguments:
```
python TrainCNN.py --data_dir /path/to/folder --train_subj 1 3 --val_subj 2 4 --epochs 10 --l_rate 0.01 --gpu_id 1
```
In the example above:
- Subjects to include in training dataset: 1 and 3
- Subjects to include in validation dataset: 2 and 4
- Number of epochs of training: 10
- Learning rate to use in training: 0.01
- ID of GPU to use in training: 1

## Evaluating the registration method

Enter the following command to evaluate the registration method:
```
python EvaluateCNN.py --weights_path /path/to/model/weights
```
By default, EvaluateCNN.py makes the following assumptions:
- Name of folder containing the test dataset: *data*
- Subject to include in test dataset: 3
- ID of GPU to use in evaluation: 0

However, these assumptions can be overridden using the following arguments:
```
python EvaluateCNN.py --weights_path /path/to.model/weights --data_dir /path/to/folder --test_subj 1 3 --gpu_id 1
```
In the example above:
- Subjects to include in test dataset: 1 and 3
- ID of GPU to use in evaluation: 1

## Weights for registration CNN of method

Weights for the registration CNN of the method will be made available soon. 