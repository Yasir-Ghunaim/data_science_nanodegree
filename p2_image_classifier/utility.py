import torch
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torch import nn, optim
from torchvision import datasets, transforms
from collections import OrderedDict
from PIL import Image

def define_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default="", required=False)
    parser.add_argument('--arch', default="vgg16", required=False)
    parser.add_argument('--learning_rate', default=0.003, required=False)
    parser.add_argument('--hidden_units', default=500, type=int, required=False)
    parser.add_argument('--epochs', default=5, type=int, required=False)
    parser.add_argument('--gpu', action='store_true', required=False)
    args = vars(parser.parse_args())
    return args

def define_predict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--gpu', action='store_true', required=False)
    parser.add_argument('--top_k', default=1, type=int, required=False)
    parser.add_argument('--category_names', default="cat_to_name.json", required=False)
    args = vars(parser.parse_args())
    return args

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_data, valid_data, test_data

def get_data_loader(dataset, isShuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=isShuffle)

def get_category_name(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_checkpoint(filename, class_to_idx, epochs, optimizer_state_dict, state_dict, args):
    directory = args["save_dir"]
    arch = args["arch"]
    hidden_units = args["hidden_units"]
    
    if directory != "":
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "/" + filename
            
    checkpoint = {'class_to_idx': class_to_idx,
                  'epochs': epochs,
                  'optimizer_state_dict': optimizer_state_dict,
                  'state_dict': state_dict,
                  'hidden_units': hidden_units,
                  'arch': arch}
    
    torch.save(checkpoint, filename)
    
def load_checkpoint(args):
    checkpoint = torch.load(args["checkpoint"])
    
    if checkpoint["arch"] == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
    
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, checkpoint["hidden_units"])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint["hidden_units"], len(checkpoint['class_to_idx']))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    # Resize to 256
    resizeFactor = 256/min(im.size)
    im.thumbnail((im.size[0]*resizeFactor, im.size[1]*resizeFactor))
    
    # Crop to 224 by 224
    width, height = im.size
    left = (width/2) - 224/2
    upper = (height/2) - 224/2
    right = (width/2) + 224/2
    lower = (height/2) + 224/2   
    im = im.crop((left, upper, right, lower))
    
    # Scale integers from 0-255 to 0-1
    np_image = np.array(im)/255
    
    # Normalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose dimensions 
    np_image = np_image.transpose(2, 0, 1)
    
    # Retrun numpy array
    return np_image
    

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (std * image) + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none') 
    ax.imshow(image)
    if title != None:
        ax.set_title(title)

    return ax

def class_num_to_name(classes, cat_to_name):
    classNames = []
    for item in classes:
        classNames.append(cat_to_name[item])
    return classNames