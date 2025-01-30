# Import Packages
import argparse
import numpy as np
import os
import json

from nas.config import *
from nas.manager import *
from models.obfuscated_model import *
from models.alexnet import alexnet

import matplotlib.pyplot as plt
from scipy import stats



def init_weights(m):

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def validate(net, data_loader, return_top5=False, device=None):
    
    print("device", device)

    net.eval()

    criterion = nn.CrossEntropyLoss()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            # compute output
            output = net(images)
            loss = criterion(output, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    
    if return_top5:
        return losses.avg, top1.avg, top5.avg
    else:
        return losses.avg, top1.avg

def fine_tuning(args, model, trainloader, testloader, opt, lr, momentum, sch, folder_path, device=None):
    
    # check if the folder exists
    os.makedirs(folder_path, exist_ok=True)

    num_epochs = 150

    criterion = nn.CrossEntropyLoss()
    
    if opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum)
    else:
        print("No optimizer")
    
    if sch == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    else:
        print("No scheduler")
    
    val_accuracy_max = 0.0
    loss_list = []
    val_loss_list = []
    val_acc_list = []
    lr_list = []
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            # Select around 10% training dataset for training
            if batch_idx > 0.1 * len(trainloader):
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

        ave_loss, acc = validate(model, testloader, device=device)
        print(f'Epoch: {epoch} | Train Loss: {loss.item():.10f} | Val Loss: {ave_loss:.10f} | Val Acc: {acc:.5f} | Lr: {optimizer.param_groups[0]["lr"]:.10f}')

        if sch is not None:
            scheduler.step()

        loss_list.append(loss.item())
        val_loss_list.append(ave_loss.item())
        val_acc_list.append(acc.item())
        lr_list.append(optimizer.param_groups[0]['lr'])
 
        if acc > val_accuracy_max:
            val_accuracy_max = acc
    
    print("Lr: {} | Max Val Acc: {}".format(lr, val_accuracy_max))
    
    # Save loss and accuracy
    loss_list = np.array(loss_list)
    val_loss_list = np.array(val_loss_list)
    val_acc_list = np.array(val_acc_list)
    lr_list = np.array(lr_list)

    if sch is not None:
        save_path = folder_path + \
                    model.__class__.__name__ + '_' + \
                    optimizer.__class__.__name__ + '_' + \
                    str(lr) + '_' + \
                    str(momentum) + \
                    '.npy'
    else:
        save_path = folder_path + \
                    model.__class__.__name__ + '_' + \
                    optimizer.__class__.__name__ + '_' + \
                    str(lr) + \
                    '.npy'
    
    np.save(save_path, (loss_list, val_loss_list, val_acc_list, lr_list))

def build_obfuscated_model(args, folder_path, device=None):

    # load network
    net_path = folder_path + 'learned_net/net.config'
    if os.path.isfile(net_path):
        with open(net_path, 'r') as f:
            net_dict = json.load(f)
        # build network
        net = AlexNet.build_from_config(net_dict)
    
    print('Network:')
    for k, v in net_dict.items():
        print('\t%s: %s' % (k, v))

    print("net", net)
    
    # Set inplace to False of all RELU in the network
    for m in net.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    # load state dict
    init_path = folder_path + 'learned_net/init'
    init = torch.load(init_path)
    
    # Load the pretrained weights
    net.load_state_dict(init['state_dict'])

    return net

def build_baseline_model(args, device=None):

    net = alexnet(num_classes=10)
    net.to(device)

    # initialize weights
    net.apply(init_weights)

    return net




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='gpu available', default='0')
    parser.add_argument('--manual_seed', default=0, type=int)
    parser.add_argument('--optim', type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--sch", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='cifar10')
    
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########################################################################################## Load Data and Run Config ##########################################################################################

    if args.dataset == 'cifar10':        
        folder_path = 'config_cifar10/'

        # load run config
        run_config_path = folder_path + 'learned_net/run.config'
        if os.path.isfile(run_config_path):
            with open(run_config_path, 'r') as f:
                run_config_dict = json.load(f)
            run_config = Cifar10RunConfig(**run_config_dict)

        # build from cifar10 data provider
        data_provider = Cifar10DataProvider(save_path='/cifar10data', 
                                            train_batch_size=256, test_batch_size=256, valid_size=None, n_worker=32, distort_color=None)

    elif args.dataset == 'cifar100':
        folder_path = 'config_cifar100/'

        # load run config
        run_config_path = folder_path + 'learned_net/run.config'
        if os.path.isfile(run_config_path):
            with open(run_config_path, 'r') as f:
                run_config_dict = json.load(f)
            run_config = Cifar100RunConfig(**run_config_dict)

        # build from cifar100 data provider
        data_provider = Cifar100DataProvider(save_path='/cifar100data', 
                                            train_batch_size=256, test_batch_size=256, valid_size=None, n_worker=32, distort_color=None)
    
    elif args.dataset == 'stl10':
        folder_path = 'config_stl10/'

        # load run config
        run_config_path = folder_path + 'learned_net/run.config'
        if os.path.isfile(run_config_path):
            with open(run_config_path, 'r') as f:
                run_config_dict = json.load(f)
            run_config = STL10RunConfig(**run_config_dict)

        # build from stl10 data provider
        data_provider = STL10DataProvider(save_path='/stl10data', 
                                            train_batch_size=256, test_batch_size=256, valid_size=None, n_worker=32, distort_color=None)
    
    else:
        print("No dataset")
    
    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    
    ################################################################# Build Obfuscated Network #################################################################
    obfuscated_net = build_obfuscated_model(args, folder_path, device=device)
    
    attack_folder_path = folder_path + 'obfuscated/'

    ################################################################## Build Baseline Network ############################################################
    baseline_net = build_baseline_model(args, device=device)

    attack_folder_path = folder_path + 'baseline/'

    ################################################################# Attack the network ################################################################
    trainloader = data_provider.train
    testloader = data_provider.test
    
    if args.optim == "SGD":
        fine_tuning(args, obfuscated_net, trainloader, testloader, "SGD", args.lr, args.momentum, args.sch, attack_folder_path, device=device)
    else:
        print("No optimizer")
