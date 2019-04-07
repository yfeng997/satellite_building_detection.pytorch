# Building classification
While classification is part of the detection module already, this module separately performs classification with better quality satellite images. 

## Data Preparation
Python script for generating training images and annotations are placed inside data/ folder. Satellite images are classified into residential vs. non-residential types. 

## Model Structure
This module uses a pre-trained DenseNet as the backbone model structure to perform binary classification. 