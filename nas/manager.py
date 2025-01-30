# Description: Manager for running the NAS algorithm

import sys
sys.path.append('../utils/')

import os
import time
import json
import copy
from datetime import timedelta

import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from nas.config import RunConfig, RLArchSearchConfig
from utils.pytorch_utils import *
from utils.model_deploy import *


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0
        
        # Initialize model
        if self.run_config.dataset == 'cifar10':
            state_dict = torch.load(
                "/state_dict/alexnet/alexnet_cifar10.pth", map_location=torch.device("cuda:0")
            )
        elif self.run_config.dataset == 'cifar100':
            state_dict = torch.load(
                "state_dict/alexnet/alexnet_cifar100.pth", map_location=torch.device("cuda:0")
            )
        elif self.run_config.dataset == 'stl10':
            state_dict = torch.load(
                "state_dict/alexnet/alexnet_stl10.pth", map_location=torch.device("cuda:0")
            )
        else:
            raise ValueError("Dataset not supported")
        self.net.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully")
        
        # Init the weights of operation in blocks with he_fout to stablize the training, prevent the exploding gradient
        for name, param in self.net.named_parameters():
            if param.requires_grad == True and "candidate_ops" in name:
                print("Name: ", name)
                torch.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        
        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.net = torch.nn.DataParallel(self.net) # Multi-GPU training
            self.net.to(self.device)
            cudnn.benchmark = True
        else:
            raise ValueError
            # self.device = torch.device('cpu')

        # net info
        self.print_net_info()

        self.criterion = nn.CrossEntropyLoss()
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_optimizer([
                self.net.module.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.net.module.get_parameters(keys, mode='include'),  # parameters without weight decay
            ])
        else:
            self.optimizer = self.run_config.build_optimizer(self.net.module.weight_parameters())

    """ save path and log path """

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path
    
    """ net info """
    def net_inference_latency(self, device=None, net=None):

        os.makedirs(self.path + "/temp/", exist_ok=True)

        # Step 1: Save the model to a temporary file
        torch.save(net, self.path + "/temp/" + 'temp_model.pth')
        # Step 2: Load the model back (this creates a deep copy)
        dup_net = torch.load(self.path + "/temp/" + 'temp_model.pth')
        dup_net = dup_net.module

        if device == "GPU":
            pass
        elif device == "CPU":
            pass
        elif device == "CPU-GPU":
            inference_latency = measure_inference_latency(dup_net, device)
        else:
            raise ValueError("Device not supported")
        
        return inference_latency

    def net_model_size(self):
        model_size = self.net.module.get_model_size()
        return model_size
    
    def print_net_info(self):
        # Network architecture
        if self.out_log:
            print(self.net)

        # Model Size
        model_size = self.net_model_size()
        if self.out_log:
            print('Model size: {}'.format(model_size))
        
        net_info = {
            'archi': str(self.net),
            'param': '%.2fM' % (model_size / 1e6),
        }

        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')
    
    """ save and load models """
    
    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'
        
        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)
        
        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print('=> loading checkpoint from %s' % model_fname)
            
            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.net.module.load_state_dict(checkpoint['state_dict'])

            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)
            
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
            
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)
        
        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)
    
    def validate(self, is_test=True, net=None, use_train_mode=False, return_top5=False):
        if is_test:
            data_loader = self.run_config.test_loader
        else:
            data_loader = self.run_config.valid_loader
        
        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
  
                if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    if return_top5:
                        test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg
    
    def train_one_epoch(self, adjust_lr_func, train_log_func):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)

            # compute output
            output = self.net(images)
            if self.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(output, labels, self.run_config.label_smoothing)
            else:
                loss = self.criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.run_config.print_frequency == 0 or i + 1 == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
                self.write_log(batch_log, 'train')

        return top1, top5
    
    def train(self, print_top5=False):
        nBatch = len(self.run_config.train_loader)

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                format(epoch_ + 1, i, nBatch - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            if print_top5:
                batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log
        
        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

            end = time.time()
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            time_per_epoch = time.time() - end
            seconds_left = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(is_test=False, return_top5=True)
                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.run_config.n_epochs, val_loss, val_acc, self.best_acc)
                if print_top5:
                    val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'.\
                        format(val_acc5, top1=train_top1, top5=train_top5)
                else:
                    val_log += '\tTrain top-1 {top1.avg:.3f}'.format(top1=train_top1)
                self.write_log(val_log, 'valid')
            else:
                is_best = False
            
            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.net.module.state_dict(),
            }, is_best=is_best)

    """ attack """
    
    def attack(self, net=None, epochs=30, stop_batch=None):
        
        ## Backup the original net's state_dict
        original_state_dict = copy.deepcopy(self.net.module.state_dict())

        ## Attack the network
        victim_net = net

        # Print the required_grad of the weight parameters
        for name, param in victim_net.named_parameters():
            print("[Attack Params BEGIN] Name: ", name, "Requires grad: ", param.requires_grad)

        # Set the Weight Parameters of the victim_net, except the blocks, to trainable
        for name, param in victim_net.named_parameters():
            if "blocks" not in name:
                param.requires_grad = True
        
        train_loader = self.run_config.attack_train_loader
        test_loader = self.run_config.test_loader
        
        # Train
        # Set the network to train mode
        victim_net.train()
        # Print the required_grad of the weight parameters
        for name, param in victim_net.named_parameters():
            print("[Attack Params MID] Name: ", name, "Requires grad: ", param.requires_grad)

        train_loss = []
        for epoch in range(epochs):            
            
            for batch, (images, labels) in enumerate(train_loader):
                
                # Select ~10% dataset for fine-tuning
                if batch == stop_batch:
                    break
                
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = victim_net(images)
                loss = self.criterion(outputs, labels)
                train_loss.append(loss.item())

                # Zero the gradients
                self.optimizer.zero_grad()
                # Check if grad_fn is None
                loss.backward()
                self.optimizer.step()

            # Validation


        # Evaluation
        # Set the network to evaluation mode
        victim_net.eval()
        
        correct = 0
        total = 0
        total_loss = 0
        total_samples = 0
        test_loss = 0
        test_acc = 0
        with torch.no_grad():

            for batch, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = victim_net(images)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                total_samples += images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = total_loss / total_samples
        test_acc = (correct / total) * 100
        print('[Attack Result] Train Loss: %.4f, Test Loss: %.4f, Test Accuracy: %.4f' % (train_loss[-1], test_loss, test_acc))

        ## Restore the original net
        net.module.load_state_dict(original_state_dict)
        # Check if two state_dicts are equal
        for key in original_state_dict.keys():
            if not torch.equal(original_state_dict[key], net.module.state_dict()[key]):
                print("Not Equal")

        # Set the Weight Parameters of the victim_net, except the blocks, back to not trainable
        for name, param in victim_net.named_parameters():
            if "blocks" not in name:
                param.requires_grad = False

        # Print the required_grad of the weight parameters
        for name, param in victim_net.named_parameters():
            print("[Attack Params END] Name: ", name, "Requires grad: ", param.requires_grad)

        # Only update the final loss and accuracy
        return train_loss[-1], test_loss, test_acc
    
    """ score """
    def score(self, net=None, x=None, target=None):

        # Store the gradient
        grads = {}
        for name, param in net.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()

        net.zero_grad()

        x.requires_grad_(True)
        y = net(x)
        temp_y = y.clone()

        y.backward(torch.ones_like(y))
        
        jacobs = x.grad.detach()
        
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

        corrs = np.corrcoef(jacobs)

        # DEBUG
        corrs_ori = copy.deepcopy(corrs)

        if np.isnan(corrs).any():
            
            # Check if there are any finite and non-NaN values
            if np.isfinite(corrs).any():
                print("------Warning------: NaN values found. Replacing NaNs with the mean of finite values.")
                # Calculate the mean of finite values
                finite_mean = np.nanmean(corrs[np.isfinite(corrs)])
                # Replace NaNs with the calculated mean
                corrs = np.where(np.isnan(corrs), finite_mean, corrs)

                # Store the jacobs and corrs and temp_y into a file
                np.save("jacobs.npy", jacobs)
                np.save("corrs.npy", corrs)
                np.save("corrs_ori.npy", corrs_ori)
                np.save("temp_y.npy", temp_y.detach().cpu().numpy())

            else:
                print("------Warning------: No finite or non-NaN values found. Using default mean:")
                # Provide a default value if no valid entries exist
                finite_mean = 0  # Default value can be adjusted based on the context
                # Replace NaNs with default value
                corrs = np.where(np.isnan(corrs), finite_mean, corrs)

                # Store the jacobs and corrs and temp_y into a file
                np.save("jacobs.npy", jacobs)
                np.save("corrs.npy", corrs)
                np.save("temp_y.npy", temp_y.detach().cpu().numpy())
        
        elif np.isinf(corrs).any():
            print("------Warning------: Inf values found. Replacing Infs with the max and min of finite values.")
            max_val = np.max(corrs[np.isfinite(corrs)])
            min_val = np.min(corrs[np.isfinite(corrs)])
            corrs = np.where(corrs == np.inf, max_val, corrs)
            corrs = np.where(corrs == -np.inf, min_val, corrs)
        
        else:
            pass
        
        v, _  = np.linalg.eig(corrs)
        v = np.where(v <= 0, 1e-5, v)
        
        k = 1e-5 # Small value to prevent division by zero

        score = -np.sum(np.log(v + k) + 1./(v + k))
        
        # Restore the gradient
        for name, param in net.named_parameters():
            if name in grads:
                param.grad = grads[name]

        return score


class ArchSearchRunManager:
    
    def __init__(self, path, super_net, run_config: RunConfig, arch_search_config: RLArchSearchConfig):
        # init weight parameters & build weight optimizer
        self.run_manager = RunManager(path, super_net, run_config, True)

        self.arch_search_config = arch_search_config

        # init architecture parameters
        self.net.init_arch_params(
            self.arch_search_config.arch_init_type, self.arch_search_config.arch_init_ratio,
        )

        # build architecture optimizer
        self.arch_optimizer = self.arch_search_config.build_optimizer(self.net.architecture_parameters())

        self.warmup = True
        self.warmup_epoch = 0
    
    @property
    def net(self):
        # The module is used to access the network in the DataParallel wrapper
        return self.run_manager.net.module
    
    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush() # Flush the data to disk, ensure the data is written immediately
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        
        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))
        
        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict']) # update the model_dict using the checkpoint
        self.net.load_state_dict(model_dict)
        if self.run_manager.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))
        
        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

        if 'epoch' in checkpoint:
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
        if 'weight_optimizer' in checkpoint:
            self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        if 'arch_optimizer' in checkpoint:
            self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        if 'warmup' in checkpoint:
            self.warmup = checkpoint['warmup']
        if self.warmup and 'warmup_epoch' in checkpoint:
            self.warmup_epoch = checkpoint['warmup_epoch']
    
    """ training related methods """
    
    def validate(self):
        # get performances of current chosen network on validation set
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # set chosen op active
        self.net.set_chosen_op_active()
        # remove unused modules
        self.net.unused_modules_off()

        # test on validation set under train mode
        valid_res = self.run_manager.validate(is_test=False, use_train_mode=False, return_top5=True)

        # measure model size
        model_size = self.run_manager.net_model_size()
        
        # unused modules back
        self.net.unused_modules_back()

        return valid_res, model_size
    
    def warm_up(self, warmup_epochs=25):
        lr_max = 0.05
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch

        for epoch in range(self.warmup_epoch, warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            
            # Switch to train mode
            self.run_manager.net.train()
            
            # Freeze the original network weights
            self.net.freeze_params()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                ## lr for sgd
                # T_cur = epoch * nBatch + i
                # warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                # for param_group in self.run_manager.optimizer.param_groups:
                #     param_group['lr'] = warmup_lr
                images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                # compute output
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup

                # forward (DataParallel)
                output = self.run_manager.net(images)

                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    loss = self.run_manager.criterion(output, labels)
                loss = -loss # maximize the acc
                print("Classification Loss: ", loss)

                # Add distriubtion loss
                regulation_loss = 0.0
                for name, module in self.net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        if module.weight.requires_grad:
                            print("Module: ", name, "Requires grad: ", module.weight.requires_grad)
                            # Print the max and min value of the weight
                            print("Before Backpropagation Max: {}, Min: {}".format(torch.max(module.weight), torch.min(module.weight)))

                            # Get the "conv1" weight parameters
                            conv_layer = getattr(self.net, 'conv1')
                            regulation_loss += L2NormRegularizer(conv_layer, module, 10) # Pay attention to the strength of regularization tuning
                print("Regulation Loss: ", regulation_loss)
                loss += regulation_loss
                
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # compute gradient and do optimizer step
                # Check if have a grad_fn for the loss, Identity function does not have a grad_fn
                if loss.grad_fn is not None:
                    # zero grads of weight_param, arch_param & binary_param
                    self.run_manager.net.zero_grad()
                    loss.backward()
                    self.run_manager.optimizer.step()  # update weight parameters
                
                # unused modules back
                self.net.unused_modules_back()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\t' \
                                'lr {lr:.5f}\t' \
                                'Classification Loss: {loss:.4f}\t' \
                                'Regulation Loss: {regulation_loss:.4f}' .\
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=self.run_manager.optimizer.param_groups[0]['lr'], loss=loss, regulation_loss=regulation_loss)
                    self.run_manager.write_log(batch_log, 'train')

            # validate
            valid_res, model_size = self.validate()
            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                        'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\tmodel_size: {5:.1f}'. \
                        format(epoch + 1, warmup_epochs, *valid_res, model_size, top1=top1, top5=top5)
            self.run_manager.write_log(val_log, 'valid')

            self.warmup = epoch + 1 < warmup_epochs
            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {
                'state_dict': state_dict,
                'warmup': self.warmup,
            }
            if self.warmup:
                checkpoint['warmup_epoch'] = epoch,
            self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')

    def train(self, fix_net_weights=False):

        torch.autograd.set_detect_anomaly(True)
        
        # Print the dataset name
        print("Dataset: ", self.run_manager.run_config.dataset)

        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch
        
        arch_param_num = len(list(self.net.architecture_parameters()))
        binary_gates_num = len(list(self.net.binary_gates()))
        weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
            (arch_param_num, binary_gates_num, weight_param_num)
        )
        
        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        print("Update Schedule: ", update_schedule)
        
        print("Epochs: ", self.run_manager.run_config.n_epochs)
        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()
            
            # Freeze the original network weights
            self.net.freeze_params()
            
            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                # network entropy
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters if not fix_net_weights
                if not fix_net_weights:
                    images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                    # compute output
                    self.net.reset_binary_gates()  # random sample binary gates
                    self.net.unused_modules_off()  # remove unused module for speedup
                    output = self.run_manager.net(images)  # forward (DataParallel)
                    # loss
                    if self.run_manager.run_config.label_smoothing > 0:
                        loss = cross_entropy_with_label_smoothing(
                            output, labels, self.run_manager.run_config.label_smoothing
                        )
                    else:
                        loss = self.run_manager.criterion(output, labels)
                    loss = -loss  # maximize the acc
                    
                    # Add distriubtion loss
                    regulation_loss = 0.0
                    # TODO Rewrite this part
                    for name, module in self.net.named_modules():
                        if isinstance(module, nn.Conv2d):
                            if module.weight.requires_grad:
                                
                                # Split the module name to get the block number
                                block_num = int(name.split(".")[1])
                                if block_num in range(0, 3):
                                    conv_layer = getattr(self.net, 'conv1_0')
                                    regulation_loss += L2NormRegularizer(conv_layer, module, 10)     
                                if block_num in range(3, 6):
                                    conv_layer = getattr(self.net, 'conv1_1')
                                    regulation_loss += L2NormRegularizer(conv_layer, module, 10)   
                                
                                if block_num in range(6, 9):
                                    conv_layer = getattr(self.net, 'conv2_0')
                                    regulation_loss += L2NormRegularizer(conv_layer, module, 10)
                                if block_num in range(9, 12):
                                    conv_layer = getattr(self.net, 'conv2_1')
                                    regulation_loss += L2NormRegularizer(conv_layer, module, 10)

                                if block_num in range(12, 15):
                                    conv_layer = getattr(self.net, 'conv3_0')
                                    regulation_loss += L2NormRegularizer(conv_layer, module, 10)
                                if block_num in range(15, 18):
                                    conv_layer = getattr(self.net, 'conv3_1')
                                    regulation_loss += L2NormRegularizer(conv_layer, module, 10)
                                
                                # if block_num in range(18, 21):
                                #     conv_layer = getattr(self.net, 'conv4_0')
                                #     regulation_loss += L2NormRegularizer(conv_layer, module, 10)
                                # if block_num in range(21, 24):
                                #     conv_layer = getattr(self.net, 'conv4_1')
                                #     regulation_loss += L2NormRegularizer(conv_layer, module, 10)
                                
                                # if block_num in range(24, 27):
                                #     conv_layer = getattr(self.net, 'conv5_0')
                                #     regulation_loss += L2NormRegularizer(conv_layer, module, 10)
                                # if block_num in range(27, 30):
                                #     conv_layer = getattr(self.net, 'conv5_1')
                                #     regulation_loss += L2NormRegularizer(conv_layer, module, 10)
                    
                    print("Regulation Loss: ", regulation_loss)
                    loss += regulation_loss

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(loss, images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # compute gradient and do SGD step
                    # Check if have a grad_fn for the loss, Identity function does not have a grad_fn
                    if loss.grad_fn is not None:
                        # zero grads of weight_param, arch_param & binary_param
                        self.run_manager.net.zero_grad()
                        loss.backward()
                        self.run_manager.optimizer.step()  # update weight parameters
                    
                    # unused modules back
                    self.net.unused_modules_back()
                
                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    # update architecture parameters according to update_schedule
                    # the update_scheule list is all 1.
                    for j in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        # RL Search
                        reward_list, net_info_list = self.rl_update_step()
                        used_time = time.time() - start_time
                        log_str = 'REINFORCE [%d-%d]\tTime %.4f\tMean Reward %.4f\t%s' % (
                                epoch + 1, i, used_time, sum(reward_list) / len(reward_list), net_info_list
                        )
                        self.write_log(log_str, prefix='rl', should_print=False)
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # training log
                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1, top5=top5, lr=self.run_manager.optimizer.param_groups[0]['lr'])
                    self.run_manager.write_log(batch_log, 'train')
            
            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.net.blocks):
                self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), model_size = self.validate()
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                          'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                          'Entropy {entropy.val:.5f}\t' \
                          'Model_Size: {6:.2f}'. \
                    format(epoch + 1, self.run_manager.run_config.n_epochs, val_loss, val_top1, self.run_manager.best_acc, val_top5,
                           model_size, entropy=entropy, top1=top1, top5=top5)
                self.run_manager.write_log(val_log, 'valid')
            # save model
            self.run_manager.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })
        
        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        os.makedirs(os.path.join(self.run_manager.path, 'learned_net'), exist_ok=True)
        json.dump(normal_net.config, open(os.path.join(self.run_manager.path, 'learned_net/net.config'), 'w'), indent=4)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, 'learned_net/run.config'), 'w'), indent=4,
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, 'learned_net/init')
        )
    
    def rl_update_step(self):
        assert isinstance(self.arch_search_config, RLArchSearchConfig)
        
        # prepare data
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False
        
        # switch to train mode
        self.run_manager.net.train()
        
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
        
        print("Batch Size: ", self.arch_search_config.batch_size)

        # sample nets and get their validation accuracy, latency, etc
        grad_buffer = []
        reward_buffer = []
        net_info_buffer = []
        for i in range(self.arch_search_config.batch_size):
            self.net.reset_binary_gates()  # random sample binary gates -> Calculate the log_prob
            self.net.unused_modules_off()  # remove unused module for speedup

            # Calculate the score of the network
            score = self.run_manager.score(self.run_manager.net, images, labels)
            print("Score: ", score)

            # get the inference latency of the network between CPU and GPU
            latency = self.run_manager.net_inference_latency(device="CPU-GPU", net=self.run_manager.net)
            
            net_info = {'score': score,
                        'latency': latency}
            net_info_buffer.append(net_info)

            # calculate reward according to net_info
            # Acc Reward
            reward = self.arch_search_config.calculate_reward_score(net_info)
            
            # loss term
            obj_term = 0
            for m in self.net.redundant_modules:
                if m.AP_path_alpha.grad is not None: # architecture parameters
                    m.AP_path_alpha.grad.data.zero_() 
                obj_term = obj_term + m.log_prob
            
            # Why negative: maximize the log distribution, which is equivalent to minimizing the negative log distribution. 
            # Help to choose the optimal operation.
            loss_term = -obj_term 
            # backward: only calculate the gradient of architecture parameters
            loss_term.backward()
            
            # take out gradient dict
            grad_list = []
            for m in self.net.redundant_modules:
                grad_list.append(m.AP_path_alpha.grad.data.clone())
            grad_buffer.append(grad_list)
            reward_buffer.append(reward)
            
            # unused modules back
            self.net.unused_modules_back()
        
        # update baseline function
        avg_reward = sum(reward_buffer) / self.arch_search_config.batch_size
        if self.arch_search_config.baseline is None:
            self.arch_search_config.baseline = avg_reward
        else:
            self.arch_search_config.baseline += self.arch_search_config.baseline_decay_weight * \
                                                (avg_reward - self.arch_search_config.baseline)
        
        # assign gradients
        for idx, m in enumerate(self.net.redundant_modules):
            m.AP_path_alpha.grad.data.zero_() # zero the gradient
            for j in range(self.arch_search_config.batch_size):
                m.AP_path_alpha.grad.data += (reward_buffer[j] - self.arch_search_config.baseline) * grad_buffer[j][idx]
            m.AP_path_alpha.grad.data /= self.arch_search_config.batch_size
        
        # apply gradients
        self.arch_optimizer.step()
        
        return reward_buffer, net_info_buffer
