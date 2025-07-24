#!/home/bsft21/tinloklee2/miniconda3/envs/depth/bin/python
import os
import math
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms_edge
from config import training_root, testing_root, backbone_path, encoder_path, ckpt_path, exp_name
from dataset_edge import ImageFolder
from dataloader import get_features, get_dataloaders
from misc import AvgMeter, check_mkdir
import loss as L

from model.dnet import DNet
from model.dnet import EncoderNet

import random
import numpy as np

def setup_directories(ckpt_path, exp_name):
    check_mkdir(ckpt_path)
    exp_dir = os.path.join(ckpt_path, exp_name)
    check_mkdir(exp_dir)
    vis_path = os.path.join(exp_dir, 'log')
    check_mkdir(vis_path)
    log_path = os.path.join(exp_dir, '/public/tinloklee2/dnet-pmd-train-log-20250713.txt')
    val_log_path = os.path.join(exp_dir, '/public/tinloklee2/dnet-pmd-val-log-20250713.txt')
    return exp_dir, log_path, val_log_path

def bce2d_new_weights(input, target):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return weights

def build_optimizer(net, args):
    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
    ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    return optimizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # est. 1 epoch takes 30 min; 
    args = {
        'epoch_num': 50, # original MSD: 100; original PMD: 160
        'batch_size': 6, # original: 10
        'last_epoch': 0,
        'lr': 1e-3,
        'lr_decay': 0.9,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'snapshot': '',
        'scale': 384,
        'seed': 0,
        'add_graph': False,
        'poly_train': True,
        'optimizer': 'SGD'
    }

    # random.seed(args['seed'])
    # np.random.seed(args['seed'])
    # torch.manual_seed(args['seed'])
    # torch.cuda.manual_seed_all(args['seed'])
    # cudnn.deterministic = True
    cudnn.benchmark = False

    # pretrained_net_path = f'{ckpt_path}/dnet/dnet.pth'

    exp_dir, log_path, val_log_path = setup_directories(ckpt_path, exp_name)
    
    net = DNet(backbone_path)
    # net.load_state_dict(torch.load(pretrained_net_path), strict=True)
    
    geo_encoder = EncoderNet([1,1,1,1,2])
    geo_encoder.load_state_dict(torch.load(encoder_path), strict=True)
    
    net = nn.DataParallel(net).to(device)
    net.train()
    optimizer = build_optimizer(net, args)
    
    geo_encoder = nn.DataParallel(geo_encoder).to(device)
    geo_encoder.eval()

    train_loader, val_loader = get_dataloaders(training_root, testing_root, geo_encoder,
                                               scale=args['scale'],
                                               batch_size=args['batch_size'])
    total_iter = args['epoch_num'] * len(train_loader)
    
    curr_iter = 1
    print("Starting training...")
    for epoch in range(args['last_epoch'] + 1, args['epoch_num'] + args['last_epoch'] + 1):
        loss_tracks = {
            'loss_total': AvgMeter(),
            'loss_4': AvgMeter(),
            'loss_3': AvgMeter(),
            'loss_2': AvgMeter(),
            'loss_1': AvgMeter(),
            'loss_edge': AvgMeter(),
            'loss_final': AvgMeter()
        }
        
        train_iter = tqdm(train_loader, total=len(train_loader))
        for data in train_iter:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_iter)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = base_lr

            inputs, labels, edges, features = data
            # print(inputs.shape)
            # print(features.shape)

            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            edges = torch.tensor(edges).to(device)
            
            optimizer.zero_grad()
            
            predict_4, predict_3, predict_2, predict_1, predict_edge, final_predict = net(inputs,features)
            
            loss_4 = L.lovasz_hinge(predict_4, labels)
            loss_3 = L.lovasz_hinge(predict_3, labels)
            loss_2 = L.lovasz_hinge(predict_2, labels)
            loss_1 = L.lovasz_hinge(predict_1, labels)
            
            bce_weight = bce2d_new_weights(predict_edge, edges)
            bce_criterion = nn.BCEWithLogitsLoss(weight=bce_weight).to(device)
            loss_edge = bce_criterion(predict_edge, edges) * 100
            loss_final = L.lovasz_hinge(final_predict, labels)
            
            loss = loss_4 + loss_3 + loss_2 + loss_1 + loss_edge + loss_final
            loss.backward()
            optimizer.step()
            
            loss_tracks['loss_total'].update(loss.item(), batch_size)
            loss_tracks['loss_4'].update(loss_4.item(), batch_size)
            loss_tracks['loss_3'].update(loss_3.item(), batch_size)
            loss_tracks['loss_2'].update(loss_2.item(), batch_size)
            loss_tracks['loss_1'].update(loss_1.item(), batch_size)
            loss_tracks['loss_edge'].update(loss_edge.item(), batch_size)
            loss_tracks['loss_final'].update(loss_final.item(), batch_size)
            
            train_log = "[E %3d, I %6d] [lr %.6f], [total %.5f], [L4 %.5f], [L3 %.5f], [L2 %.5f], [L1 %.5f], [edge %.5f], [final %.5f]" % (
                epoch, curr_iter%512, base_lr, loss_tracks['loss_total'].avg,
                loss_tracks['loss_4'].avg, loss_tracks['loss_3'].avg, loss_tracks['loss_2'].avg,
                loss_tracks['loss_1'].avg, loss_tracks['loss_edge'].avg, loss_tracks['loss_final'].avg)
            # train_iter.set_description(train_log)
            with open(log_path, 'a') as f:
                f.write(train_log + "\n")
            curr_iter += 1
        
        # Validation loop.
        print('Starting validation for epoch', epoch)
        net.eval()
        val_loss_tracks = {
            'loss_total': AvgMeter(),
            'loss_4': AvgMeter(),
            'loss_3': AvgMeter(),
            'loss_2': AvgMeter(),
            'loss_1': AvgMeter(),
            'loss_edge': AvgMeter(),
            'loss_final': AvgMeter()
        }
        
        val_iter = tqdm(val_loader, total=len(val_loader))
        for data in val_iter:
            inputs, labels, edges, features = data
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            edges = torch.tensor(edges).to(device)
            
            with torch.no_grad():
                outputs = net(inputs, features)
                predict_4, predict_3, predict_2, predict_1, predict_edge, final_predict = outputs
                loss_4 = L.lovasz_hinge(predict_4, labels)
                loss_3 = L.lovasz_hinge(predict_3, labels)
                loss_2 = L.lovasz_hinge(predict_2, labels)
                loss_1 = L.lovasz_hinge(predict_1, labels)
                bce_weight = bce2d_new_weights(predict_edge, edges)
                bce_criterion = nn.BCEWithLogitsLoss(weight=bce_weight).to(device)
                loss_edge = bce_criterion(predict_edge, edges) * 100
                loss_final = L.lovasz_hinge(final_predict, labels)
                loss_val = loss_4 + loss_3 + loss_2 + loss_1 + loss_edge + loss_final
            
                val_loss_tracks['loss_total'].update(loss_val.data, batch_size)
                val_loss_tracks['loss_4'].update(loss_4.data, batch_size)
                val_loss_tracks['loss_3'].update(loss_3.data, batch_size)
                val_loss_tracks['loss_2'].update(loss_2.data, batch_size)
                val_loss_tracks['loss_1'].update(loss_1.data, batch_size)
                val_loss_tracks['loss_final'].update(loss_final.data, batch_size)
                val_loss_tracks['loss_edge'].update(loss_edge, batch_size)

                val_log = '[%3d], [Total: %.5f], [L4: %.5f], [L3: %.5f], [L2: %.5f], [L1: %.5f], [edge: %.5f], [final_l: %.5f]' % \
                  (epoch, val_loss_tracks['loss_total'].avg, val_loss_tracks['loss_4'].avg, val_loss_tracks['loss_3'].avg, val_loss_tracks['loss_2'].avg,
                   val_loss_tracks['loss_1'].avg, val_loss_tracks['loss_edge'].avg, val_loss_tracks['loss_final'].avg)
            # val_iter.set_description(val_log)
            with open(val_log_path, 'a') as f:
                f.write(val_log + "\n")
        net.train()

        if epoch >= args['epoch_num'] or epoch % 10 == 0:
            print("Saving checkpoint at epoch ", epoch)
            net.cpu()
            # torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            torch.save(net.module.state_dict(), os.path.join('/public/tinloklee2/dnet_pmd_%d.pth' % epoch))
            if epoch >= args['epoch_num']:
                print("Training finished!")
                return
            net.to(device)

if __name__ == '__main__':
    main()