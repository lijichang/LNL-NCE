import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import normalize
import math
import glob, os, shutil
import torchnet

def export_pyfile(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for ext in ('py', 'pyproj', 'sln'):
        for fn in glob.glob('*.' + ext):
            shutil.copy2(fn, target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src', '*.' + ext)):
                shutil.copy2(fn, target_dir)

def create_folder_and_save_pyfile(nowtime, args):
    root_folder = os.path.join("./", "outputs", nowtime+"-"+'%s_%.2f_%s' % (args.dataset, args.r, args.noise_mode))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    # saving pyfiles
    folder = os.path.join(root_folder, "folder_for_pyfiles")
    if not os.path.exists(folder):
        os.makedirs(folder)
    export_pyfile(folder)
    return root_folder

class NegEntropy(object):
    def __call__(self, outputs):
        probs = F.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)

def js_div(p, q):
    # Jensen-Shannon divergence, value is in range (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def mixup(inputs, targets, alpha):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)
    idx = torch.randperm(inputs.size(0))
    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    return mixed_input, mixed_target

def eval_train_nce(args, eval_loader, net, feat_dim, num_class):
    net.eval()
    ## Get train features
    trainFeatures = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    trainLogits = torch.rand(len(eval_loader.dataset), num_class).t().cuda()
    trainNoisyLabels = torch.rand(len(eval_loader.dataset)).cuda()

    iter_count = 0
    for batch_idx, (inputs, labels, index) in enumerate(eval_loader):
        batchSize = inputs.size(0)
        logits, features = net(inputs.cuda(), feat=True)
        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.cuda().data.t()
        trainLogits[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = logits.cuda().data.t()
        trainNoisyLabels[batch_idx * batchSize:batch_idx * batchSize + batchSize] = labels.cuda().data
        iter_count += 1

    trainFeatures = normalize(trainFeatures.t())
    trainLogits = trainLogits.t()
    trainNoisyLabels = trainNoisyLabels

    # caculating neighborhood-based label inconsistency score
    num_batch = math.ceil(float(trainFeatures.size(0)) / args.batch_size) # 391
    sver_collection = []
    for batch_idx in range(num_batch):
        features = trainFeatures[batch_idx * args.batch_size:batch_idx * args.batch_size + args.batch_size]
        noisy_labels = trainNoisyLabels[batch_idx * args.batch_size:batch_idx * args.batch_size + args.batch_size]
        dist = torch.mm(features, trainFeatures.t())
        dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  # set self-contrastive samples to -1
        _, neighbors = dist.topk(args.num_neighbor, dim=1, largest=True, sorted=True) # find contrastive neighbors
        neighbors = neighbors.view(-1)
        neigh_logits = trainLogits[neighbors]
        neigh_probs = F.softmax(neigh_logits, dim=-1)
        M, _ = features.shape
        given_labels = torch.full(size=(M, num_class), fill_value=0.0001).cuda()
        given_labels.scatter_(dim=1, index=torch.unsqueeze(noisy_labels.long(), dim=1), value=1 - 0.0001)
        given_labels = given_labels.repeat(1, args.num_neighbor).view(-1, num_class)
        sver = js_div(neigh_probs, given_labels)
        sver_collection += sver.view(-1, args.num_neighbor).mean(dim=1).cpu().numpy().tolist()
    prob = np.array(sver_collection)
    # prob = 1.0 - np.array(sver_collection)
    mask_lab = prob < args.threshold_sver
    mask_unl = prob > args.threshold_sver

    labeledFeatures = trainFeatures[mask_lab]
    labeledLogits = trainLogits[mask_lab]
    labeledNoisyLabels = trainNoisyLabels[mask_lab]
    labeledW = prob[mask_lab]

    knn_labeledLogits = labeledLogits[labeledW > 0.95]
    knn_labeledFeatures = labeledFeatures[labeledW > 0.95]
    knn_labeledNoisyLabels = labeledNoisyLabels[labeledW > 0.95]

    unlabeledFeatures = trainFeatures[mask_unl]
    unlabeledLogits = trainLogits[mask_unl]
 
    num_labeled = knn_labeledFeatures.size(0)
    num_unlabeled = unlabeledFeatures.size(0)
    if num_labeled <= args.num_neighbor * num_class:
        pseudo_labels = [-3] * num_unlabeled
        pseudo_labels = np.array(pseudo_labels)
        print("num_labeled <= args.num_neighbor * 10 ...")
        return prob, noisy_labels

    # caculating pseudo-labels for unlabeled samples
    num_batch_unlabeled = math.ceil(float(unlabeledFeatures.size(0)) / args.batch_size)
    pseudo_labels = []
    scor_collection = []
    for batch_idx in range(num_batch_unlabeled):
        features = unlabeledFeatures[batch_idx * args.batch_size:batch_idx * args.batch_size + args.batch_size]
        logits = unlabeledLogits[batch_idx * args.batch_size:batch_idx * args.batch_size + args.batch_size]
        dist = torch.mm(features, knn_labeledFeatures.t())
        _, neighbors = dist.topk(args.num_neighbor, dim=1, largest=True, sorted=True) # find contrastive neighbors
        neighbors = neighbors.view(-1)
        neighs_labels = knn_labeledNoisyLabels[neighbors]
        neighs_logits = knn_labeledLogits[neighbors]
        neigh_probs = F.softmax(neighs_logits, dim=-1)
        neighbor_labels = torch.full(size=neigh_probs.size(), fill_value=0.0001).cuda()
        neighbor_labels.scatter_(dim=1, index=torch.unsqueeze(neighs_labels.long(), dim=1), value=1 - 0.0001)
        scor = js_div(F.softmax(logits.repeat(1, args.num_neighbor).view(-1, num_class), dim=-1), neighbor_labels)
        w = (1 - scor).type(torch.FloatTensor)
        w = w.view(-1, 1).type(torch.FloatTensor).cuda()
        neighbor_labels = (neighbor_labels * w).view(-1, args.num_neighbor, num_class).sum(dim=1)
        pseudo_labels += neighbor_labels.cpu().numpy().tolist()
        scor = scor.view(-1, args.num_neighbor).mean(dim=1)
        scor_collection += scor.cpu().numpy().tolist()
    scor_collection = np.array(scor_collection)

    pseudo_labels = np.argmax(np.array(pseudo_labels), axis=1)
    pseudo_labels[np.equal(scor_collection > args.threshold_scor, scor_collection <= args.high_scor)] = -1
    pseudo_labels[scor_collection > args.high_scor] = -2

    noisy_labels = trainNoisyLabels.cpu().numpy()
    noisy_labels[mask_unl>0] = pseudo_labels

    return prob, noisy_labels

def warmup(epoch, net, optimizer, dataloader, CEloss, log):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
 
        # penalty = conf_penalty(outputs)
        L = loss  # + penalty
        L.backward()
        optimizer.step()

        log.write('\r')
        log.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f' % (args.id, epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        log.flush()

# Training
def train(args, epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, log='record.txt'):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, _, labels_x, w_x, _) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, inputs_us, _, _, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_us, _, _, _ = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)
        
        # transforming given label to one-hot vector for labeled samples
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.long().view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_us = inputs_u.cuda(), inputs_u2.cuda(), inputs_us.cuda()

        # label refinement (refer to DivideMix)
        with torch.no_grad():
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
        all_targets = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a[:batch_size * 2] + (1 - l) * input_b[:batch_size * 2]
        mixed_target = l * target_a[:batch_size * 2] + (1 - l) * target_b[:batch_size * 2]
        mixed_logits = net(mixed_input)

        # mixup regularization for labeled data
        Lx = -torch.mean(torch.sum(F.log_softmax(mixed_logits, dim=1) * mixed_target, dim=1))

        # penalty regularization for mixed labeled data
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(mixed_logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # overall loss
        loss = Lx + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.write('\r')
        log.write('Webvision | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Lx loss: %.4f, Lpen: %.4f'% (epoch, args.num_epochs, batch_idx + 1, num_iter,
                        Lx.item(), penalty.item()))
        log.flush()

def test(net1, net2, test_loader):
    net1.eval()
    net2.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2

            _, predicted = torch.max(outputs, 1)
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)
            
            correct += predicted.eq(targets).cpu().sum().item()
            correct1 += predicted1.eq(targets).cpu().sum().item()
            correct2 += predicted2.eq(targets).cpu().sum().item()
            
            total += targets.size(0)

    acc = 100. * correct / total
    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total

    return acc, acc1, acc2
