import argparse
import os
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def validate_parameters():
    print("validate_parameters")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but GPU unavailable")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    data_dir = os.listdir(args.data_directory)
    if (('test' not in data_dir) or ('train' not in data_dir) or ('valid' not in data_dir)):
        raise Exception('missing: one of the sub-directories')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Select vgg or densenet')    
             
def transform_data(data_dir):
    print("transform_data")
    train_dir, test_dir, valid_dir = data_dir 
    train_transforms = transforms.Compose([
                              transforms.RandomRotation(45),
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])

    test_valid_transforms = transforms.Compose([
                              transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])    
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    train_image_loaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=24, shuffle=True)
    valid_image_loaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=24, shuffle=True)
    test_image_loaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=24, shuffle=True)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train':train_image_loaders,'valid':valid_image_loaders,'test':test_image_loaders,'labels':cat_to_name ,'train_datasets':train_image_datasets}
    return loaders

def retrieve_data():
    print("retrieve_data")
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    data_dir = [train_dir,test_dir,valid_dir]
    return transform_data(data_dir)

def build_model(data):
    print("build_model")
    
    arch_type = args.arch
    if (arch_type is None):
        arch_type = 'vgg'
        
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node=25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node=1024
        
    hidden_units = args.hidden_units
    if (hidden_units is None):
        hidden_units = 4096
        
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_node, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    return model

def test_accuracy(model,loader,device='cpu'):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total  

def train(model,data):
    print("train")
    
    print_every=40
    
    learn_rate = args.learning_rate
    if (learn_rate is None):
        learn_rate = 0.001
        
    epochs = args.epochs
    if (epochs is None):
        epochs = 10
        
    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
    
    learn_rate = float(learn_rate)
    epochs = int(epochs)
    
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    steps = 0
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()     
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_accuracy = test_accuracy(model,validloader,device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))            
                running_loss = 0
    print("FINISHED TRAINING!")
    test_result = test_accuracy(model,testloader,device)
    print('Final result - accuracy on test set: {}'.format(test_result))
    return model

def save_model(model, data):
    print("saving model")
    
    save_dir = args.save_dir
    if (save_dir is None):
        save_dir = 'checkpoint.pth'
    train_image_datasets = data['train_datasets']
    model.class_to_idx = train_image_datasets.class_to_idx
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, save_dir)
    return 0

def create_model():
    validate_parameters()
    data = retrieve_data()
    model = build_model(data)
    model = train(model,data)
    save_model(model,data)
    return None

def parse():
    parser = argparse.ArgumentParser(description='Training a neural network')
    parser.add_argument('data_directory', help='data directory')
    parser.add_argument('--save_dir', help='directory to save network.')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='enable gpu mode')
    args = parser.parse_args()
    return args

def main():
    print("Training and creating a deep learning model")
    global args
    args = parse()
    create_model()
    print("Done!!!")
    return None

main()

