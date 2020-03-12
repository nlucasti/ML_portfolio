import argparse
# Imports here
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import json


parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('image_path', metavar='d', type=str,
                    help='path of the image')
parser.add_argument('checkpoint', type = str,metavar='c', action='store',
                    help='directory where checkpoint is located')
parser.add_argument('--top_k', dest='top_k', 
                    type = int,metavar='tk', action='store',
                    help='top number of class probabilities')
parser.add_argument('--category_names', type=str, dest='category_names',
                    metavar='cn',action='store',
                    help='category names to map from')
parser.add_argument('--gpu', dest='gpu',action='store_true',
                    help='number of epochs to run')
args = parser.parse_args()

# Use GPU if it's available
device = torch.device("cuda" if args.gpu else "cpu")

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen



def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_orig_size = (256,256)
    img_new_size = 224
    # TODO: Process a PIL image for use in a PyTorch model
    img.thumbnail(img_orig_size,Image.ANTIALIAS)
    width, height = img.size
    left = (width - img_new_size)/2
    upper = (height - img_new_size)/2
    right = (width + img_new_size)/2
    lower = (height + img_new_size)/2
    
    img = img.crop((left, upper, right, lower))
    np_img = np.array(img)
    np_img = np_img/255.0
    
    
    np_img = normalize(np_img)
    final_img = np_img.transpose((2,0,1))#torch.from_numpy(np_img.transpose((2,0,1))).type(torch.cuda.FloatTensor)
    #final_img = final_img.unsqueeze_(0)
    return final_img

def normalize(img):
    r_mean, g_mean, b_mean = [0.485, 0.456, 0.406]
    r_std, g_std, b_std = [0.229, 0.224, 0.225]
    r_img = img[:,:,0]
    g_img = img[:,:,1]
    b_img = img[:,:,2]
    norm_img = np.stack(((r_img-r_mean)/r_std,
                         (g_img-g_mean)/g_std,
                         (b_img-b_mean)/b_std), axis = 2) 
    return(norm_img)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = Image.open(image_path)
    inputs = process_image(img)
    inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor).unsqueeze_(0)
    logps = model.forward(inputs)
    ps = torch.exp(logps)
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    top_p, top_class = ps.topk(topk, dim=1)

    mapped_class = [idx_to_class[x] for x in top_class.cpu().tolist()[0]]


    return (top_p, mapped_class, img)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = eval(f'models.{checkpoint.get("model")}(pretrained=True)')
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device);
    return model

model = load_checkpoint(args.checkpoint)

probs, classes, img = predict(args.image_path,model,args.top_k)
if(args.category_names != None):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    mapped_classes = [cat_to_name[x] for x in classes]
    classes = mapped_classes
probs = probs.detach().cpu().numpy().tolist()[0]
print(probs, classes)
for x in range(len(classes)):   
    print('Flower: ', classes[x])
    print('Probability: ', probs[x])