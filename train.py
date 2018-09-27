import argparse
import numpy as np
from PIL import Image

from torchvision import datasets, transforms, models 
import torch
from torch import nn, optim

from collections import OrderedDict 

from os import listdir

def get_inp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='flowers/train',
                        help='path to training set directory')
    parser.add_argument('--valid_dir', type=str, default='flowers/valid', 
                        help='path to validation dataset')
    parser.add_argument('--test_dir', type=str, default='flowers/test', 
                        help='path to testing dataset')
    parser.add_argument('--arch', type=str, default = 'resnet101', 
                        choices = ['inception', 'resnet101', 'densenet121'],
                        help='neural network architecture or model to train')
    parser.add_argument('--optim', type=str, default = 'adam', 
                        choices = ['sgd', 'rms_prop', 'adam'],
                        help='algorithm used to minimize the loss function and optimize model performance')
    parser.add_argument('--lr', type=float, default = .001, help='learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default = 35, help='number of training epochs')
    parser.add_argument('--hidden_units', type=int, default = 200, help = 'number of hidden units in the head classifier')
    parser.add_argument('--output_units', type=int, default = 102, help='number of classes to classify and output')
    parser.add_argument('--drop_p', type=int, default=0.5, help = 'proportion of units dropped or knocked out for dropout')
    parser.add_argument('--device', type=str, default = 'cuda', choices = ['cuda', 'cpu'], 
                        help='device to train on: either cuda for gpu or cpu')
    parser.add_argument('--labels', type=str, default = 'cat_to_name.json', 
                        help='file with flower names that correspond to class numbers')
    parser.add_argument('--checkpoint', type=str, default = 'checkpoint.pth', 
                        help='path to pre-trained resnet50 trained on flower dataset')
    return parser.parse_args()

def load_data(train_dir, valid_dir, test_dir):
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), 
                                       transforms.RandomRotation(35),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                      ])

    val_test_transforms = transforms.Compose([transforms.Resize(255), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                         ])

    # Loads the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # Uses the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader

resnet101 = models.resnet101(pretrained=True)
densenet121 = models.densenet121(pretrained=True)
inceptionV3 = models.inception_v3(pretrained=True)

models = {'resnet101':resnet101, 'densenet':densenet121, 'inception':inceptionV3}

def get_optimizer(arch, lr):
    
    if (arch == 'resnet101' or arch == 'inception'):
        sgd = optim.SGD(models['resnet101'].fc.parameters(), lr = lr)
        adam = optim.Adam(models['resnet101'].fc.parameters(), lr = lr)
        rms_prop = optim.RMSprop(models['resnet101'].fc.parameters(), lr = lr)
    elif arch == 'densenet121':
        sgd = optim.SGD(models['densenet121'].classifier.parameters(), lr = lr)
        adam = optim.Adam(models['densenet121'].classifier.parameters(), lr = lr)
        rms_prop = optim.RMSprop(models['densenet121'].classifier.parameters(), lr = lr)
        
    return sgd, adam, rms_prop 

def head_classifier(arch, hidden_units, output_units, drop_p):
    model = models[arch]
    if arch == 'resnet101':
        model.fc = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(2048, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('dropout', nn.Dropout(p=drop_p)),
                                    ('fc2', nn.Linear(hidden_units, output_units)), 
                                    ('output', nn.LogSoftmax(dim=1))
                                            ]))
    elif arch == 'densenet121':
        model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(1024, hidden_units1)),
                                    ('relu', nn.ReLU()),
                                    ('dropout', nn.Dropout(p=drop_p)),
                                    ('fc2', nn.Linear(hidden_units, output_units)), 
                                    ('output', nn.LogSoftmax(dim=1))
                                                         ]))
    elif arch == 'inception':
        model.fc = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(2048, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('dropout', nn.Dropout(p=drop_p)),
                                    ('fc2', nn.Linear(hidden_units, output_units)), 
                                    ('output', nn.LogSoftmax(dim=1))
                                            ]))
    return model 

def validation(loader, device='cpu'):
    
    model.to(device)
    model.eval() #turns dropout off 
    
    criterion = nn.NLLLoss()
    running_loss = 0
    acc = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            running_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            eq = (labels.data == ps.max(dim=1)[1])
            acc += eq.type(torch.FloatTensor).mean()
    
    return running_loss/len(loader), acc/len(loader) * 100

def train(model, trainloader, validloader, epochs, optimizer, device='cpu'):   
    epochs = epochs 
    
    steps = 0 
    running_loss = 0
    acc = 0 
    
    model.to(device)
    
    criterion = nn.NLLLoss()#Negative Log Likelihood Loss function

    for e in range(epochs):
        
        for idx, (images, labels) in enumerate(trainloader):
            
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            ps = torch.exp(output)
            eq = (labels.data == ps.max(dim=1)[1])
            acc += eq.type(torch.FloatTensor).mean()

            if steps % 40 == 0:
                
                val_loss, val_acc = validation(validloader, device)
                
                print(
                     'Epoch {}/{}'.format(e+1, epochs), 
                     'training loss: {:.4f}'.format(running_loss/40), 
                     'training accuracy: {:.2f}%  '.format(acc/40 * 100), 
                     'Validation loss: {:.4f}'.format(val_loss), 
                     'Validation accuracy: {:.2f}%'.format(val_acc)
                     )
                
                running_loss = 0
                acc = 0
               

inp = get_inp_args()
trainloader, validloader, testloader = load_data(inp.train_dir, inp.valid_dir, inp.test_dir)
for param in models[inp.arch].parameters(): 
    param.requires_grad = False
model = head_classifier(inp.arch, inp.hidden_units, inp.output_units, inp.drop_p)
sgd, adam, rms_prop = get_optimizer(inp.arch, inp.lr)
opt_dict = {'sgd':sgd, 'adam':adam, 'rms_prop':rms_prop}
optimizer = opt_dict[inp.optim]
train(model, trainloader, validloader, inp.epochs, optimizer)


