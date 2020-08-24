from __future__ import print_function

import logging 
import datetime
import sys
import random
import os
import torch
import torchvision
import argparse
import yaml
import time
import shutil

import torch.nn as nn
import torch.optim as optim

from dataloader.data_factory import get_dataset
#from model.model_factory import get_model
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
from model.mlp import MLP

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

best_acc = 0
use_cuda = torch.cuda.is_available()


def main():
  device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
  
  global best_acc
  start_epoch = 0

  if not os.path.isdir(args['checkpoint']):
      mkdir_p(args['checkpoint'])

  tfboard_dir = os.path.join(args['checkpoint'], 'tfboard')
  if not os.path.isdir(tfboard_dir):
    os.makedirs(tfboard_dir)
  writer = SummaryWriter(tfboard_dir)
  
  #visda number of classes
  args['num_classes'] = 12

  num_classes = args['num_classes']
  in_features = num_classes
  num_domains = 2
  num_source_domains = 1
  num_target_domains = 1 

  #dataset loading
  train_dataset = get_dataset("{}_{}".format('train', 'train')) 
  train_dataloader = data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], drop_last=True, pin_memory=True)
  
  val_dataset = get_dataset("{}_{}".format('validation', 'val'))
  val_dataloader = data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True) 

  #model loading
  model = MLP(args['model_name'], args['num_classes'])
  model = model.cuda(args['gpu'])
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])

  #resume
  title = 'visda'+args['model_name']
  if args['resume']:
      args['checkpoint'] = os.path.dirname(args['resume'])
      checkpoint =  torch.load(args['resume'])
      best_acc = checkpoint['best_acc']
      start_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
  else:
      logger = Logger(os.path.join(args['checkpoint'], 'log.txt'), title=title)
      logger.set_names(['Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
#train and val
  for epoch in range(start_epoch, args['epochs']):
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, epoch, use_cuda)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)

        test_loss, test_acc = test(val_dataloader, model, criterion, epoch, use_cuda)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

        # append logger file
        logger.append([train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args['checkpoint'])



  logger.close()
  logger.plot()
  savefig(os.path.join(args['checkpoint'], 'log.eps'))

  print('Best acc:')
  print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))



if __name__=='__main__':
  main()
