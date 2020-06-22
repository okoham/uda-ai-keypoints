import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor

# load a model, load some images, predict keypoints, calculate error

data_transform = transforms.Compose([
    Rescale((224,224)),
    #RandomCrop(),
    Normalize(),
    ToTensor(),
])

# the training data: grayscale image, keypoints
# torch.Size([1, 224, 224]) torch.Size([68, 2])
train_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)

test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                             root_dir='data/test/',
                                             transform=data_transform)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=False, 
                          num_workers=4)

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=False, 
                          num_workers=4)


# Error
# 1. take a fixed batch (potentially all) of images from training (or test) data
# 2. predict keypoints
# 3. compute error (average euclidean distance, per image)

# Visualise
# 1. take a batch of images
# 2. predict keypoints
# 3. overlay true and predicted keypoints on image
# 4. visualize images and errors as a grid
# 5. save that image


def predict_batch(dataset, model, n):
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts
            

if __name__ == '__main__':
    pass