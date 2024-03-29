import os
import torch
import wandb
import random
import argparse
import torchvision

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import *
from cifarmodels import *

def dumpData(flag, string):
    if flag == 'train':
        meta = open(args.name+'/metadataTrain.txt', "a")
        meta.write(string)
        meta.close()
    else:
        meta = open(args.name+'/metadataTest.txt', "a")
        meta.write(string)
        meta.close()

def train(net, trainloader, criterion, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    string = str(epoch) + ',' + str(train_loss) + ',' + str(correct*1.0/total) + '\n'
    dumpData('train', string)
    wandb.log({
        "Train Loss": train_loss,
        "Train Accuracy": 100*correct/total}, step=epoch)

def test(net, testloader, criterion, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    string = str(epoch) + ',' + str(test_loss) + ',' + str(correct*1.0/total) + '\n'
    dumpData('test', string)
    wandb.log({
        "Test Loss": test_loss,
        "Test Accuracy": 100*correct/total}, step=epoch)
    return correct*1.0/total

def main():
    wandb.init(name=args.name, project="cifar-results")
    # config = wandb.config
    # config.batch_size = 128
    # config.epochs = 100
    # config.lr = 0.1
    # config.momentum = 0.9           
    
    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.Dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
        numClasses = 10
    elif args.Dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transform_test)
        numClasses = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    if args.baseline == True:
        if args.Network == 'ResNet':
            net = ResNet50(numClasses)
        elif args.Network == 'MobileNet':
            net = MobileNetV2(numClasses)
        elif args.Network == 'EfficientNet':
            net = EfficientNetB0(numClasses)
        elif args.Network == 'SqueezeNet':
            net = SqueezeNet(numClasses)
    else:
        if args.Network == 'ResNet':
            net = ResNet50Friendly(numClasses)
        elif args.Network == 'MobileNet':
            net = MobileNetV2Friendly(numClasses)
        elif args.Network == 'EfficientNet':
            net = EfficientNetB0Friendly(numClasses)
        elif args.Network == 'SqueezeNet':
            net = SqueezeNetFriendly(numClasses)
    
    criterion = nn.CrossEntropyLoss().cuda()    
    optimizer = torch.optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)

    net.cuda()
    wandb.watch(net, log="all")
    bestAcc = 0
    startEpoch = 0
    if args.resume == True:
        assert os.path.isdir(args.name), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.name+'/BestModel.t7')
        net.load_state_dict(checkpoint['net'])
        bestAcc = checkpoint['acc']
        startEpoch = checkpoint['epoch'] 
        optimizer.load_state_dict(checkpoint['optimizer'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[20, 40, 60, 80], gamma=0.1, last_epoch=startEpoch-1)
    
    for epoch in range(startEpoch, 100):
        train(net, trainloader, criterion, optimizer, epoch)
        lr_scheduler.step()
        acc = test(net, testloader, criterion, epoch)
        state = {'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch+1,
                'optimizer' : optimizer.state_dict()
                }
        if acc > bestAcc:
            torch.save(state, args.name+'/BestModel.t7')
            bestAcc = acc
            wandb.save('BestModel.h5')
        else:
            torch.save(state, args.name+'/LastEpoch.t7')
    
    meta = open(args.name+'/stats.txt', "a")
    s = 'baseline' if args.baseline==True else 'friendly' 
    meta.write(args.Dataset + ' , ' + args.Network + ' , ' + s + ' , ' + str(bestAcc) + '\n')    
    meta.close()

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser(description = "Train CIFAR Models")
    parser.add_argument("--Dataset", "-D", type = str, help = 'CIFAR10, CIFAR100', required=True)
    parser.add_argument("--Network", "-N", type = str, help = 'ResNet, MobileNet, EfficientNet, SqueezeNet', required=True)
    parser.add_argument("--name", "-n", type=str, help = 'Name of the run', required=True)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--baseline', '-b', action='store_true', help='Baseline or Friendly')
    args = parser.parse_args()

    if not os.path.isdir(args.name):
        os.mkdir(args.name)

    main()
