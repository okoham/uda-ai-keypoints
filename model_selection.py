import sys
import time
import copy 


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch import optim
#import torchvision
#from torchvision import models
# create dataloader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import Subset

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import models


DEVICE = torch.device("cpu")



def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """
    dataloaders: dict, with kays "train", "val"
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf 

    dataset_sizes = {k: len(loader.dataset) for k, loader in dataloaders.items()}

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
            #running_corrects = 0

            # Iterate over data.
            dataloader = dataloaders[phase]
            for batch_i, data in enumerate(dataloader):

                # get the input images and their corresponding labels
                images = data['image']
                key_pts = data['keypoints']

                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to floats for regression loss
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

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

                # statistics
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


    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def _test_model(model, device, dataloader, prefix=""):
    """Evaluate the model. Can be used for validation (during training) and testing.
    
    Params: 
    - model: the model
    - dataloader: a torch.utils.data.DataLoader instance
    - prefix: What to display on screen during evaluation (usually "Validation" or "Test"
    
    Returns:
    - loss
    - accuracy
    """

    model.eval()   # Set model to evaluate mode

    dataset_size = len(dataloader.dataset)
    count = 0
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device) # shape (batchsize, 3, width, height)
        labels = labels.to(device) # 1-d array, shape (batchsize,)
        # forward
        with torch.no_grad():
            outputs = model(inputs) # shape: 
            proba = torch.exp(outputs) # shape: 
            preds = torch.argmax(proba, dim=1) # 1-d array, shape (batchsize,)
            loss = criterion(outputs, labels) 
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        count += inputs.size(0)
        # display progress
        sys.stdout.write("\r%s: %i/%i - loss: %.3f, acc: %.3f" % (prefix, count, 
                                                 dataset_size, 
                                                 running_loss/count, 
                                                 running_corrects.double()/count))
    sys.stdout.write("\n")
    
    loss = running_loss / dataset_size
    acc = running_corrects.item() / dataset_size
    return loss, acc



def _train_model(model, device, optimizer, train_dataloader, valid_dataloader, num_epochs=5):
    """Train a model.
    
    Parameters:
    - model: a model instance
    - optimizer: an optimizer instance
    - train_dataloader: 
    - valid_dataloader: 
    - num_epochs: number of epochs to train (default 5)
    
    Returns:
    - None
    """
    training_size = len(train_dataloader.dataset)
    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        running_loss = 0.0             # running training loss
        running_corrects = 0           # running training correct samples
        count = 0                      # running number of samples
        
        # Iterate over data batches
        for inputs, labels in train_dataloader: 
            inputs = inputs.to(device)
            labels = labels.to(device) # 1-d array, shape (batchsize,)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            proba = torch.exp(outputs)
            preds = torch.argmax(proba, dim=1) # 1-d array, shape (batchsize,)
            loss = criterion(outputs, labels)
            # backward 
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            count += inputs.size(0)
            sys.stdout.write("\rTrain: %i/%i - loss: %.3f, acc: %.3f" % (count, 
                                                     training_size, 
                                                     running_loss/count, 
                                                     running_corrects.double()/count))
                
        epoch_loss = running_loss / training_size
        epoch_acc = running_corrects.double() / training_size
        print()
        # validation 
        model.eval()
        valid_loss, valid_acc = test_model(model, device, valid_dataloader, prefix="Validation")
        print('Train loss: {:.4f}, acc: {:.4f}    Valid loss: {:.4f}, acc: {:.4f}\n'.format(epoch_loss, epoch_acc, valid_loss, valid_acc))
        model.train()


    
def save_checkpoint(model, fpath, arch, n_hidden, class_to_idx, optimizer, epochs, lr, description=""):
    """Save a model alsong with some hyperparameters.
    
    Params:
    - model: the model instance to save
    - fpath (string): path where the model should be saved
    - arch (string): the model architechture (like 'vgg16', vgg16_bn', ...)
    - n_hidden (int): number of hidden units in classifier layer
    - class_to_idx (dict of str: int): distionary, maps class names (directory names) to index of output layer
    - optimizer: the optimizer instance used to train the model.
    - epochs (int): number of epochs trained so far
    - lr (float): learning rate used
    - desription (str): some descriptive information, e.g. how well it performed om test data
    
    Returns:
    - None
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": model.arch,
        "n_hidden": model.n_hidden,
        "class_to_idx": class_to_idx,
        "description": description,
        "epochs": epochs,
        "lr": lr,
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, fpath)
        
        
        
def load_checkpoint(fpath):
    """Return a saved model.
    
    Params:
    - fpath (str): path to model checkpoint file
    
    Returns: 
    - model
    """
    
    checkpoint = torch.load(fpath, map_location='cpu')
    print(checkpoint["description"])
    
    arch = checkpoint['arch']
    class_to_idx = checkpoint['class_to_idx']
    # build empty model (without pretrained weights), then load the saved weights.
    model = build_model(arch, len(class_to_idx), n_hidden=checkpoint['n_hidden'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    # attach class names and inverted index
    model.class_to_idx = class_to_idx
    model.idx_to_class = {v:k for k, v in class_to_idx.items()}
    model.eval()
    
    return model
    

def history_plot_image(history, img_path):
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


if __name__ == '__main__':

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

    # FIXME
    train_val_dataset = Subset(train_val_dataset, range(500))

    print('Number of images: ', len(train_val_dataset))

    # split train/val
    # FIXME: make a function of ot this
    pct_train = 0.7
    batch_size = 16

    import random
    random.seed(1234)

    num_samples = len(train_val_dataset)
    num_train = int(pct_train*num_samples)
    #rng = np.random.default_rng(1234)
    indices = np.arange(num_samples)
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(train_val_dataset, train_indices)
    val_dataset = Subset(train_val_dataset, val_indices)

    dataloaders = {'train': DataLoader(train_dataset, 
                                       batch_size=batch_size,
                                       shuffle=True, 
                                       num_workers=4),
                   'val': DataLoader(val_dataset, 
                                     batch_size=batch_size,
                                     shuffle=False, 
                                     num_workers=4),                                       
    }



    model = models.Net12()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()


    model, history = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5)

    # store model, store history
    fig = history_plot_image(history, 'runs/daddel1.png')



    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/experiment_3')
    #writer.add_image('history', img)

    model_dir = 'saved_models/'
    model_name = 'keypoints_model_xx.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(model.state_dict(), model_dir+model_name)

    writer.add_figure('train_val', fig)
    #writer.add_graph(model)
    writer.add_hparams({'batch_size': 16,
                        'lr': 0.01,
                        'arch': 'Net12',
                       })
    writer.add_scalars('lalala', history)
    writer.close()