from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import argparse
from InceptionResNetV2 import *
import dataloader_webvision as dataloader
import datetime
import time
from utils import create_folder_and_save_pyfile
from utils import warmup, train, test
from utils import eval_train_nce
from utils import NegEntropy

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='natrual')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=120, type=int)
parser.add_argument('--warm_up', default=1, type=int)
parser.add_argument('--r', default=0.0, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='./data/webvision/', type=str, help='path to dataset')
parser.add_argument('--dataset', default='webvision', type=str)
parser.add_argument('--remark', default='dividemix', type=str)
parser.add_argument('--feat_dim', default=1536, type=int)
parser.add_argument('--num_neighbor', default=20, type=int)
parser.add_argument('--threshold_sver', default=0.90, type=float)
parser.add_argument('--threshold_scor', default=0.05, type=float)
parser.add_argument('--high_scor', default=1.0, type=float)
args = parser.parse_args()

run_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
root_folder = create_folder_and_save_pyfile(run_time + "-" + args.remark, args)
log = open(os.path.join(root_folder, '%s_%.2f_%s' % (args.dataset, args.r, args.noise_mode) + '_records.txt'), 'a+')
record_log = open(os.path.join(root_folder, '%s_%.2f_%s' % (args.dataset, args.r, args.noise_mode) + '_record.txt'), 'a+')
ils_test_log = open(os.path.join(root_folder, '%s_%.2f_%s' % (args.dataset, args.r, args.noise_mode) + '_imgnet_acc.txt'), 'a+')
web_test_log = open(os.path.join(root_folder, '%s_%.2f_%s' % (args.dataset, args.r, args.noise_mode) + '_web_acc.txt'), 'a+')

print('| Building net')
def create_model():
    model = InceptionResNetV2(num_classes=args.num_class)
    model = nn.DataParallel(model)
    model = model.cuda()
    return model
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

print('| Building optimizer')
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, num_class=args.num_class)
web_valloader = loader.run('test')
imagenet_valloader = loader.run('imagenet')

web_acc1, web_acc2, web_acc3 = test(net1, net2, web_valloader)
web_test_log.write('Epoch:%d \t WebVision Acc-NET1+NET2: %.2f%% (%.2f%%) \tNET1: %.2f%% (%.2f%%) \tNET2: %.2f%% (%.2f%%)\n' % (
        -1, web_acc3[0], web_acc3[1], web_acc1[0], web_acc1[1], web_acc2[0], web_acc2[1]))
web_test_log.flush()

ign_acc1, ign_acc2, ign_acc3 = test(net1, net2, imagenet_valloader)
ils_test_log.write('Epoch:%d \t ILSVRC12 Acc-NET1+NET2: %.2f%% (%.2f%%) \tNET1: %.2f%% (%.2f%%) \tNET2: %.2f%% (%.2f%%)\n' % (
        -1, ign_acc3[0], ign_acc3[1], ign_acc1[0], ign_acc1[1], ign_acc2[0], ign_acc2[1]))
ils_test_log.flush()

for epoch in range(args.num_epochs + 1):
    start_time = time.time()
    lr = args.lr
    if epoch >= 40 and epoch < 80:
        lr /= 10
    elif epoch >= 80:
        lr /= 10 * 10

    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    
    warmup_trainloader = loader.run('warmup')
    eval_loader = loader.run('eval_train')

    # model training
    if epoch < args.warm_up:
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader, CEloss, log)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader, CEloss, log)

    else:
        pred1 = (prob1 < args.threshold_sver)
        pred2 = (prob2 < args.threshold_sver)
        log.write('Epoch:%d \tLAB-NET1:%d\tNET2:%d \n' % (epoch, pred1.sum(), pred2.sum()))
        log.write('Epoch:%d \tUNL-NET1:%d\tNET2:%d \n' % (epoch, len(pred1) - pred1.sum(), len(pred2) - pred2.sum()))
        log.flush()

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, 1-prob2, lab2)  # co-divide
        train(args, epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, log=record_log)

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, 1-prob1, lab1)  # co-divide
        train(args, epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, log=record_log)

    web_acc1, web_acc2, web_acc3 = test(net1, net2, web_valloader)
    web_test_log.write('Epoch:%d \t WebVision Acc-NET1+NET2: %.2f%% (%.2f%%) \tNET1: %.2f%% (%.2f%%) \tNET2: %.2f%% (%.2f%%)\n' % (
            epoch, web_acc3[0], web_acc3[1], web_acc1[0], web_acc1[1], web_acc2[0], web_acc2[1]))
    web_test_log.flush()

    ign_acc1, ign_acc2, ign_acc3 = test(net1, net2, imagenet_valloader)
    ils_test_log.write('Epoch:%d \t ILSVRC12 Acc-NET1+NET2: %.2f%% (%.2f%%) \tNET1: %.2f%% (%.2f%%) \tNET2: %.2f%% (%.2f%%)\n' % (
            epoch, ign_acc3[0], ign_acc3[1], ign_acc1[0], ign_acc1[1], ign_acc2[0], ign_acc2[1]))
    ils_test_log.flush()

    prob1, lab1 = eval_train_nce(args, eval_loader, net1, feat_dim=args.feat_dim, num_class=args.num_class)
    prob2, lab2 = eval_train_nce(args, eval_loader, net2, feat_dim=args.feat_dim, num_class=args.num_class)


