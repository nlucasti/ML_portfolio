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


parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('data_directory', metavar='d', type=str,
                    help='path of the data')
parser.add_argument('--save', dest='save_directory', type = str,metavar='s', action='store',
                    help='directory where checkpoint will be saved')
parser.add_argument('--arch', dest='architecture', default="vgg19",
                    type = str,metavar='a', action='store',
                    help='architecture of the model to be created')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default =0.003,
                    metavar='lr',action='store',
                    help='learning rate of model')
parser.add_argument('--hidden_units', type = int, dest='hidden_units', 
                    default=512, metavar='hu',action='store',
                    help='hidden units in the model layer')
parser.add_argument('--epochs', type = int, default = 7, dest='epochs',metavar='e', action='store',
                    help='number of epochs to run')
parser.add_argument('--gpu', dest='gpu',action='store_true',
                    help='number of epochs to run')
args = parser.parse_args()

model = eval(f'models.{args.architecture}(pretrained=True)')

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    #Define classifier
classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),  
                           nn.Linear(args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
        #model =models.{args.model}(pretrained=True)
        #device = torch.device(arg.gpu
device = torch.device("cuda" if args.gpu else "cpu")



# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {'training':transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                   'validation': transforms.Compose([transforms.Resize(224),
                                                     transforms.RandomCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



# TODO: Load the datasets with ImageFolder
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'

image_datasets = {'training':datasets.ImageFolder(train_dir, transform = data_transforms['training']),
                  'validation':datasets.ImageFolder(valid_dir, transform = data_transforms['validation'])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'training':torch.utils.data.DataLoader(image_datasets['training'], batch_size=128, shuffle = True),
              'validation':torch.utils.data.DataLoader(image_datasets['validation'], batch_size=128)}

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device);

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

from workspace_utils import active_session
#### PICK UP HERE
with active_session():
    for epoch in range(epochs):
        for inputs, labels in dataloaders['training']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['validation']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(dataloaders['validation']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['validation']):.3f}")
                running_loss = 0
                model.train()

model.class_to_idx = image_datasets['training'].class_to_idx
checkpoint = {'class_to_idx': model.class_to_idx,
                'output_size': args.hidden_units,
              'epochs': args.epochs,
              'state_dict': model.state_dict(),
             'optimizer': optimizer,
             'model': args.architecture,
             'classifier': classifier}
if(args.save_directory != None):
    torch.save(checkpoint, args.save_directory)