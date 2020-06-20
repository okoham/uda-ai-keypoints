import sys
import time
import copy 
import datetime

import numpy as np
import random
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import Subset

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

import models


DEVICE = torch.device("cpu")
random.seed(1234)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=1):
    """Train a model. After training, the model weights are set to those that resulted 
    in the lowest validation loss during training.
    
    Parameters:
    - model: a model instance
    - dataloaders: a dictionary. Must have keys "train" and "val", and dataloaders as values
    - criterion: a loss function
    - optimizer: an optimizer instance
    - scheduler: a method from torch.optim.lr_scheduler
    - num_epochs: number of epochs to train (default 1)
    
    Returns:
    - history: dict with training/validation loss history
    """

    best_model_wts = copy.deepcopy(model.state_dict())

    # set initial loss to infinity
    best_loss = np.inf 

    dataset_sizes = {k: len(loader.dataset) for k, loader in dataloaders.items()}

    # initialise 
    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data batches
            dataloader = dataloaders[phase]
            for batch_i, data in enumerate(dataloader):

                # get the input images and their corresponding labels
                images = data['image']
                key_pts = data['keypoints']

                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to floats for regression loss
                images = images.type(torch.FloatTensor)
                key_pts = key_pts.type(torch.FloatTensor)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()                
                
                # forward pass to get outputs
                output_pts = model.forward(images)

                # calculate the loss between predicted and target keypoints
                loss = criterion(output_pts, key_pts)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()            

                # Statistics. Loss function returns mean loss over images; multiply by size of current batch
                running_loss += loss.item() * images.size(0)

            # learning rate scheduler
            if phase == 'train':
                scheduler.step()

            # epoch finished
            epoch_loss = running_loss / dataset_sizes[phase]
            history[phase].append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
          
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # final message
    print('\nBest val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return history

    

def plot_history(history):
    """history plot, return image"""
    fig, ax = plt.subplots()
    ax.plot(history['train'], "o-", label="train")
    ax.plot(history['val'], "o-", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()

    # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    #fig.canvas.draw()
    #image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    #return image_from_plot
    #fig.savefig(img_path)
    return fig


def train_val_split(dataset, train_size=0.8):
    """Split dataset into training and validation part. Data are shuffled.
    """

    assert train_size <= 1

    num_samples = len(dataset)       
    num_train = int(train_size*num_samples)

    indices = np.arange(num_samples)
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset



if __name__ == '__main__':

    import argparse

    archs = ['Net12', 'Net12d', 'Net22', 'Net22d', 'Net32d', 'Net33d']
    optimizers = ['Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'SGD']
    loss_funcs = {'mse_loss': F.mse_loss, 'l1_loss': F.l1_loss, 'smooth_l1_loss': F.smooth_l1_loss}

    parser = argparse.ArgumentParser(description='Train a new model.')
    #parser.add_argument('--title', type=str, default="", 
    #                    help='short title of model')
    parser.add_argument('--subset_size', type=int, default=0, 
                        help='Use only a subset of data (for development)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (default 0.001)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='number of epochs for training (default 10)')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='batch size (default 10)')
    parser.add_argument('--arch', type=str, default='Net12', 
                        help='Network architecture, one of: %s. Default Net12' % archs)
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        help='Optimizers, one of: %s. Default Net12' % optimizers)
    parser.add_argument('--loss_func', type=str, default='mse_loss', 
                        help='Loss function, one of: %s. Default mse_loss' % list(loss_funcs.keys()))
    parser.add_argument('--save_dir', type=str, default="runs", 
                        help='path to directory where trained model checkpoints should be saved')

    args = parser.parse_args()

    assert args.arch in archs, f"Error: Network architecture {args.arch} not found."
    arch = getattr(models, args.arch)
    model = arch()

    assert args.optimizer in optimizers, f"Error: Optimizer {args.optimizer} not found."
    opti = getattr(optim, args.optimizer)
    optimizer = opti(model.parameters(), lr=args.lr)

    assert args.loss_func in loss_funcs, f"Error: Network architecture {args.loss_func} not found."
    criterion = loss_funcs[args.loss_func]

    # hard-coded
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    

    # Define a transform
    data_transform = transforms.Compose([
        Rescale((224,224)),
        #RandomCrop(),
        Normalize(),
        ToTensor(),
    ])

    # create the transformed dataset
    train_val_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                root_dir='data/training/',
                                                transform=data_transform)

    # for faster experimentation select only a subset of images
    if args.subset_size > 0:
        train_val_dataset = Subset(train_val_dataset, range(args.subset_size))
    
    print('Number of images: ', len(train_val_dataset))

    # split train/val - hard coded
    train_size = 0.8
    train_dataset, val_dataset = train_val_split(train_val_dataset, train_size=train_size)

    # create dataloaders
    dataloaders = {'train': DataLoader(train_dataset, 
                                       batch_size=args.batch_size,
                                       shuffle=True, 
                                       num_workers=4),
                   'val': DataLoader(val_dataset, 
                                     batch_size=args.batch_size,
                                     shuffle=False, 
                                     num_workers=4),                                       
    }


    t0 = time.time()
    history = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=args.epochs)
    time_elapsed = time.time() - t0
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))    

    title = datetime.datetime.now().isoformat()

    # store model, store history
    fig = plot_history(history)
    fig.savefig(f'{args.save_dir}/{title}_curves.png')

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(model.state_dict(), f"{args.save_dir}/{title}.pt")

    # save parameters. 
    params = {'loss_history': history,
            'title': title,
            'time_elapsed': time_elapsed
    }
    params.update(vars(args))


    with open(f"{args.save_dir}/{title}_params.json", "w") as fp:
        json.dump(params, fp, indent=2)