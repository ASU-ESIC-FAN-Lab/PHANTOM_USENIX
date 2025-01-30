import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
import numpy as np
import time
import argparse

from cifar10data.cifar10 import *
from cifar100data.cifar100 import *
from stl10.stl10 import *

import matplotlib.pyplot as plt


torch.autograd.set_detect_anomaly(True)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)



"""
Origianl AlexNet
"""
class AlexNetOri(nn.Module):
    def __init__(self, num_classes):
        
        super(AlexNetOri, self).__init__()
        
        self.ReLU = nn.ReLU(inplace=False)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
       
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.conv3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.ReLU(x)

        x = self.conv5(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ReLU(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ReLU(x)

        x = self.fc3(x)
        
        return x



"""
    Model Definition
    3 - 1 - 1
    5 - 1 - 2
    7 - 1 - 3
    9 - 1 - 4
"""

class AlexNet(nn.Module):

    def __init__(self, num_classes=10, layer_selection=None, ks=3, str=1, pad=1):

        super(AlexNet, self).__init__()

        self.layer_selection = layer_selection
        self.num_classes = num_classes
        
        self.ReLU = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        if layer_selection[0] == 1:
            self.conv1_extra = nn.Conv2d(64, 64, kernel_size=ks, stride=str, padding=pad)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        if layer_selection[1] == 1:
            self.conv2_extra = nn.Conv2d(192, 192, kernel_size=ks, stride=str, padding=pad)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        if layer_selection[2] == 1:
            self.conv3_extra = nn.Conv2d(384, 384, kernel_size=ks, stride=str, padding=pad)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        if layer_selection[3] == 1:
            self.conv4_extra = nn.Conv2d(256, 256, kernel_size=ks, stride=str, padding=pad)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        if layer_selection[4] == 1:
            self.conv5_extra = nn.Conv2d(256, 256, kernel_size=ks, stride=str, padding=pad)
       
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.ReLU(x)
        if self.layer_selection[0] == 1:
            x = self.conv1_extra(x)
            x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.conv2(x)
        x = self.ReLU(x)
        if self.layer_selection[1] == 1:
            x = self.conv2_extra(x)
            x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.conv3(x)
        x = self.ReLU(x)
        if self.layer_selection[2] == 1:
            x = self.conv3_extra(x)
            x = self.ReLU(x)

        x = self.conv4(x)
        x = self.ReLU(x)
        if self.layer_selection[3] == 1:
            x = self.conv4_extra(x)
            x = self.ReLU(x)

        x = self.conv5(x)
        x = self.ReLU(x)
        if self.layer_selection[4] == 1:
            x = self.conv5_extra(x)
            x = self.ReLU(x)
        x = self.MaxPool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ReLU(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ReLU(x)

        x = self.fc3(x)
        
        return x


"""
    Train Function
"""
def train(model, device, train_loader, test_loader, index):
    
    # Check the trainable layers
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_acc = 100
    for epoch in range(0, 200):

        model.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Degrade the performance
            loss = -loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss/total}, Train Accuracy: {100.*correct/total}")

        # Eval Model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch: {epoch}, Test Accuracy: {100.*correct/total}")

        if 100.*correct/total < best_acc:
            best_acc = 100.*correct/total
            torch.save(model.state_dict(), "sensitive_layer/state_dict/alexnet_cifar10_layer_" + str(index) + ".pth")
        
        scheduler.step()
    
    print("Best Accuracy: ", best_acc)
    print("Training Complete")


"""
    Fine-tuning
"""
def fune_tuning(model, device, train_loader, test_loader, index):

    # Check the trainable layers
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    best_acc = 0

    loss_list = []
    train_acc = []
    val_acc = []
    for epoch in range(0, 150):

        model.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # Train with 10% of the training dataset
            if batch_idx > 0.1 * len(train_loader):
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        loss_list.append(train_loss/total)
        train_acc.append(100.*correct/total)
        print(f"Epoch: {epoch}, Train Loss: {train_loss/total}, Train Accuracy: {100.*correct/total}")

        # Eval Model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc.append(100.*correct/total)
        print(f"Epoch: {epoch}, Test Accuracy: {100.*correct/total}")
        
        if 100.*correct/total > best_acc:
            best_acc = 100.*correct/total
            # torch.save(model.state_dict(), "sensitive_layer/fine_tune/alexnet_cifar10_layer_" + str(index) + ".pth")
            torch.save(model.state_dict(), "sensitive_layer/ori/ori.pth")

        scheduler.step()
    
    # Save the loss and accuracy to the one file
    # np.save("sensitive_layer/fine_tune/alexnet_cifar10_layer_" + str(index) + "_loss.npy", loss_list)
    # np.save("sensitive_layer/fine_tune/alexnet_cifar10_layer_" + str(index) + "_train_acc.npy", train_acc)
    # np.save("sensitive_layer/fine_tune/alexnet_cifar10_layer_" + str(index) + "_val_acc.npy", val_acc)
    np.save("sensitive_layer/ori/ori_loss.npy", loss_list)
    np.save("sensitive_layer/ori/ori_train_acc.npy", train_acc)
    np.save("sensitive_layer/ori/ori_val_acc.npy", val_acc)

    print("Best Accuracy: ", best_acc)
    print("Training Complete")


"""
    Plot
"""
def plot():
    
    # Initialize a list to store validation accuracies
    val_acc_list = []
    for index in range(5):
        val_acc_file = f"sensitive_layer/fine_tune/alexnet_cifar10_layer_{index}_val_acc.npy"
        val_acc = np.load(val_acc_file)
        print("max val_acc: ", np.max(val_acc))
        val_acc_list.append(val_acc)

    # Create a figure for plotting
    plt.figure(figsize=(10, 6))

    # Get a colormap with 20 distinct colors
    cmap = plt.get_cmap('tab10')
    
    # Plot each validation accuracy line with a different color
    for i, val_acc in enumerate(val_acc_list):
        plt.plot(val_acc, color=cmap(i / 5), label=f"Validation Accuracy Layer {i}")
    
    # baseline_acc = 75.0 
    # plt.axhline(y=baseline_acc, color='red', linestyle='--', label='Baseline Accuracy')

    # Set labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy for Different Layers')

    # Add a legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend outside the plot

    # Save the figure
    plt.savefig("sensitive_layer/fine_tune/val_acc_zoomin.png")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='gpu available', default='1')
    parser.add_argument('--dataset', help='dataset', default=None)
    args = parser.parse_args()
    
    # Set GPU
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    
    ########################################### Load dataset CIFAR10 ###########################################
    data_provider = Cifar10DataProvider("cifar10data/")
    train_loader = data_provider.train
    test_loader = data_provider.test

    print("Length of train_loader: ", len(train_loader))
    print("Length of test_loader: ", len(test_loader))

    ########################################### Layer Selection ###########################################
    layer_selection = [0, 0, 0, 0, 0]

    for index in range(0, 5):
        layer_selection[index] = 1

        print("Layer Selection: ", layer_selection)

        ########################################### Load Model ###########################################
        model = AlexNet(num_classes=10, layer_selection=layer_selection, ks=3, str=1, pad=1).to(device)

        # Load weight for training
        # state_dict = torch.load("/home/state_dict/alexnet/alexnet_cifar10_customized.pth")
        # model.load_state_dict(state_dict, strict=False)
        
        # Load weight for fine-tuning
        state_dict = torch.load("/home/alexnet/sensitive_layer/state_dict/alexnet_cifar10_layer_" + str(index) + ".pth")
        model.load_state_dict(state_dict)
        
        ########################################### Eval ###########################################
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Test Accuracy: {100.*correct/total}")

        ######################################### Freeze ###########################################
        # print the trainable layers name
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if "extra" in name:
        #             print(name)
        #         else:
        #             param.requires_grad = False

        # for module in model.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module.eval()
        
        ########################################### Train ###########################################
        # train(model, device, train_loader, test_loader, index)

        ########################################### Fine-Tuning ###########################################
        fune_tuning(model, device, train_loader, test_loader, index)

        layer_selection[index] = 0

    ########################################### Original AlexNet ###########################################
    model = AlexNetOri(num_classes=10).to(device)

    fune_tuning(model, device, train_loader, test_loader, index=None)
