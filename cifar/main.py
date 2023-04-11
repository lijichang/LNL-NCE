from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import os, datetime, argparse
from PreResNet import *
import dataloader_cifar as dataloader
from utils import NegEntropy
from utils import test
from utils import train
from utils import warmup
from utils import ncnv, nclc
from utils import create_folder_and_save_pyfile

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--lr_switch_epoch', default=150, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--drop', default=0.0, type=float)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--remark', default='', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

run_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
root_folder = create_folder_and_save_pyfile(run_time + "-" + args.remark, args)
record_log = open(os.path.join(root_folder, '%s_%.2f_%s' % (args.dataset, args.r, args.noise_mode) + '_records.txt'), 'a+')
test_log = open(os.path.join(root_folder, '%s_%.2f_%s' % (args.dataset, args.r, args.noise_mode) + '_results.txt'), 'a+')

if args.dataset == 'cifar10':
    warm_up = 10
    threshold_sver = 0.75
    threshold_scor = 0.002
    if args.noise_mode == 'asym':
        threshold_sver = 0.50
        threshold_scor = 0.0005
elif args.dataset == 'cifar100':
    warm_up = 30
    threshold_sver = 0.90
    threshold_scor = 0.01
    if args.r == 0.5:
        threshold_scor = 0.005
if args.r <= 0.2:
    threshold_scor = 0.0

print('| Building net')
def create_model(args):
    model = ResNet18(num_classes=args.num_class, drop=args.drop)
    model = model.cuda()
    return model
net1 = create_model(args)
net2 = create_model(args)
cudnn.benchmark = True

print('| Building optimizer')
optimizer1 = optim.SGD(list(net1.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(list(net2.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()
else:
    conf_penalty = None

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=5, \
                root_dir=args.data_path, noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= args.lr_switch_epoch:
        lr /= 10
    if epoch >= (args.lr_switch_epoch * 2):
        lr /= 2

    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    warmup_trainloader = loader.run('warmup')
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    # model training
    if epoch < warm_up:
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader, CEloss, args, conf_penalty)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader, CEloss, args, conf_penalty, log=record_log)

    else:
        prob1 = ncnv(net1, eval_loader, batch_size=args.batch_size, num_class=args.num_class)
        pred1 = (prob1 < threshold_sver)
        prob2 = ncnv(net2, eval_loader, batch_size=args.batch_size, num_class=args.num_class)
        pred2 = (prob2 < threshold_sver)

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, 1-prob2)  # co-divide
        pseudo_labels = nclc(net1, net2, labeled_trainloader, unlabeled_trainloader, test_loader, batch_size=args.batch_size, num_class=args.num_class, threshold_scor=threshold_scor)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, args, pseudo_labels=pseudo_labels, log=record_log)

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, 1-prob1)  # co-divide
        pseudo_labels = nclc(net2, net1, labeled_trainloader, unlabeled_trainloader, test_loader, batch_size=args.batch_size, num_class=args.num_class, threshold_scor=threshold_scor)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, args, pseudo_labels=pseudo_labels, log=record_log)

    # model testing
    test(epoch, net1, net2, test_log, test_loader)




