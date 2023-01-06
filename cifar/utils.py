import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import normalize
import math
import glob, os, shutil

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

def getFeature(net, net2, trainloader, testloader, feat_dim, num_class):
    transform_bak = trainloader.dataset.transform
    trainloader.dataset.transform = testloader.dataset.transform
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=8)

    trainFeatures = torch.rand(len(trainloader.dataset), feat_dim).t().cuda()
    trainLogits = torch.rand(len(trainloader.dataset), num_class).t().cuda()
    trainW = torch.rand(len(trainloader.dataset)).cuda()
    trainNoisyLabels = torch.rand(len(trainloader.dataset)).cuda()

    for batch_idx, (inputs, _, _, labels, _, w, _) in enumerate(temploader):
        batchSize = inputs.size(0)
        logits, features = net(inputs.cuda(), feat=True)
        logits2, features2 = net2(inputs.cuda(), feat=True)

        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = (features+features2).data.t()
        trainLogits[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = (logits+logits2).data.t()
        trainNoisyLabels[batch_idx * batchSize:batch_idx * batchSize + batchSize] = labels.cuda().data
        trainW[batch_idx * batchSize:batch_idx * batchSize + batchSize] = w.data

    trainFeatures = trainFeatures.detach().cpu().numpy()
    trainLogits = trainLogits.detach().cpu().numpy()
    trainNoisyLabels = trainNoisyLabels.detach().cpu().numpy()
    trainW = trainW.detach().cpu().numpy()

    trainloader.dataset.transform = transform_bak
    return (trainFeatures, trainLogits, trainNoisyLabels, trainW)

## function for Neighborhood Collective Noise Verification (NCNV step)
def ncnv(net, eval_loader, num_class, batch_size, feat_dim=512, num_neighbor=20):
    net.eval()

    # loading given samples
    trainFeatures = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    trainLogits = torch.rand(len(eval_loader.dataset), num_class).t().cuda()
    trainNoisyLabels = torch.rand(len(eval_loader.dataset)).cuda()
    for batch_idx, (inputs, labels, _, _) in enumerate(eval_loader):
        batchSize = inputs.size(0)
        logits, features = net(inputs.cuda(), feat=True)
        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
        trainLogits[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = logits.data.t()
        trainNoisyLabels[batch_idx * batchSize:batch_idx * batchSize + batchSize] = labels.cuda().data

    trainFeatures = normalize(trainFeatures.t())
    trainLogits  = trainLogits.t()
    trainNoisyLabels = trainNoisyLabels

    # caculating neighborhood-based label inconsistency score
    num_batch = math.ceil(float(trainFeatures.size(0)) / batch_size) # 391
    sver_collection = []
    for batch_idx in range(num_batch):
        features = trainFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        noisy_labels = trainNoisyLabels[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(features, trainFeatures.t())
        dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  # set self-contrastive samples to -1
        _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True) # find contrastive neighbors
        neighbors = neighbors.view(-1)
        neigh_logits = trainLogits[neighbors]
        neigh_probs = F.softmax(neigh_logits, dim=-1)
        M, _ = features.shape
        given_labels = torch.full(size=(M, num_class), fill_value=0.0001).cuda()
        given_labels.scatter_(dim=1, index=torch.unsqueeze(noisy_labels.long(), dim=1), value=1 - 0.0001)
        given_labels = given_labels.repeat(1, num_neighbor).view(-1, num_class)
        sver = js_div(neigh_probs, given_labels)
        sver_collection += sver.view(-1, num_neighbor).mean(dim=1).cpu().numpy().tolist()
    sver_collection = np.array(sver_collection)
    return sver_collection

## function for Neighborhood Collective Label Correction (NCLC step)
def nclc(net, net2, labeled_trainloader, unlabeled_trainloader, testloader, threshold_scor, batch_size, num_class, feat_dim=512, num_neighbor=20, high_scor=1.0):
    net.eval()
    net2.eval()

    # loading labeled samples
    labeledFeatures, labeledLogits, labeledNoisyLabels, labeledW = getFeature(net, net2, labeled_trainloader, testloader, feat_dim, num_class)
    knn_labeledLogits = labeledLogits.T[labeledW > 0.5]
    knn_labeledFeatures = labeledFeatures.T[labeledW > 0.5]
    knn_labeledNoisyLabels = labeledNoisyLabels[labeledW > 0.5]
    knn_labeledLogits = torch.from_numpy(knn_labeledLogits).cuda()
    knn_labeledFeatures = torch.from_numpy(knn_labeledFeatures).cuda()
    knn_labeledNoisyLabels = torch.from_numpy(knn_labeledNoisyLabels).cuda()

    # loading unlabeled samples
    unlabeledFeatures, unlabeledLogits, _, _ = getFeature(net, net2, unlabeled_trainloader, testloader, feat_dim, num_class)
    unlabeledFeatures = torch.from_numpy(unlabeledFeatures.T).cuda()
    unlabeledLogits = torch.from_numpy(unlabeledLogits.T).cuda()

    # normalizing features
    knn_labeledFeatures = normalize(knn_labeledFeatures)
    unlabeledFeatures = normalize(unlabeledFeatures)

    num_labeled = knn_labeledFeatures.size(0)
    num_unlabeled = unlabeledFeatures.size(0)
    if num_labeled <= num_neighbor * 10:
        pseudo_labels = [-3] * num_unlabeled
        pseudo_labels = np.array(pseudo_labels)
        print("num_labeled <= num_neighbor * 10 ...")
        return torch.from_numpy(pseudo_labels)
 
    # caculating pseudo-labels for unlabeled samples
    num_batch_unlabeled = math.ceil(float(unlabeledFeatures.size(0)) / batch_size)
    pseudo_labels = []
    scor_collection = []
    for batch_idx in range(num_batch_unlabeled):
        features = unlabeledFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        logits = unlabeledLogits[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(features, knn_labeledFeatures.t())
        _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True) # find contrastive neighbors
        neighbors = neighbors.view(-1)
        neighs_labels = knn_labeledNoisyLabels[neighbors]
        neighs_logits = knn_labeledLogits[neighbors]
        neigh_probs = F.softmax(neighs_logits, dim=-1)
        neighbor_labels = torch.full(size=neigh_probs.size(), fill_value=0.0001).cuda()
        neighbor_labels.scatter_(dim=1, index=torch.unsqueeze(neighs_labels.long(), dim=1), value=1 - 0.0001)
        scor = js_div(F.softmax(logits.repeat(1, num_neighbor).view(-1, num_class), dim=-1), neighbor_labels)
        w = (1 - scor).type(torch.FloatTensor)
        w = w.view(-1, 1).type(torch.FloatTensor).cuda()
        neighbor_labels = (neighbor_labels * w).view(-1, num_neighbor, num_class).sum(dim=1)
        pseudo_labels += neighbor_labels.cpu().numpy().tolist()
        scor = scor.view(-1, num_neighbor).mean(dim=1)
        scor_collection += scor.cpu().numpy().tolist()
    scor_collection = np.array(scor_collection)

    pseudo_labels = np.argmax(np.array(pseudo_labels), axis=1)
    pseudo_labels[np.equal(scor_collection > threshold_scor, scor_collection <= high_scor)] = -1
    pseudo_labels[scor_collection > high_scor] = -2

    return torch.from_numpy(pseudo_labels)

def warmup(epoch, net,  optimizer, dataloader, CEloss, args, conf_penalty, log):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, gt_labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode == 'sym':
            L = loss
        L.backward()
        optimizer.step()

        if (batch_idx + 1) % 50 == 0:
            log.write('\r')
            log.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                             % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                                loss.item()))
            log.flush()

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, args, pseudo_labels, log):
    net.train()
    net2.eval()  # fix one network and train the other

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    num_iter_labeled = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    num_iter_unlabeled = (len(unlabeled_trainloader.dataset) // args.batch_size) + 1
    num_iter = max(num_iter_labeled, num_iter_unlabeled)

    for batch_idx in range(num_iter):
        try:
            inputs_xw, inputs_xw2, inputs_xs, labels_x, _, w_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_xw, inputs_xw2, inputs_xs, labels_x, _, w_x, _ = labeled_train_iter.next()

        try:
            inputs_uw, inputs_uw2, inputs_us, labels_u, _, _, index_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_uw, inputs_uw2, inputs_us, labels_u, _, _, index_u = unlabeled_train_iter.next()

        # transforming given label to one-hot vector for labeled samples
        targets_x = torch.zeros(inputs_xw.size(0), args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        targets_x = targets_x.cuda()
        labels_x = labels_x.long().cuda()
        mask_x = labels_x >= 0

        # assigning corrected pseudo-labels for unlabeled samples
        labels_u = labels_u.long().cuda()
        labels_u_temp = pseudo_labels[index_u].long().cuda()
        mask_u = labels_u_temp >= 0
        labels_u[mask_u] = labels_u_temp[mask_u]

        inputs_xw = inputs_xw.cuda()
        inputs_xw2 = inputs_xw2.cuda()
        inputs_xs = inputs_xs.cuda()
        inputs_uw = inputs_uw.cuda()
        inputs_us = inputs_us.cuda()

        # label refinement (refer to DivideMix)
        with torch.no_grad(): 
            outputs_x = net(inputs_xw)
            outputs_x2 = net(inputs_xw2)
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            w_x = w_x.view(-1, 1).type(torch.FloatTensor).cuda()
            px = w_x * targets_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        mixed_inputs = torch.cat([inputs_xw, inputs_xw2], dim=0)
        mixed_targets = torch.cat([targets_x, targets_x], dim=0)
        mixed_input, mixed_target = mixup(mixed_inputs, mixed_targets, alpha=args.alpha)
        mixed_logits = net(mixed_input)

        # mixup regularization for labeled data
        Lx = -torch.mean(torch.sum(F.log_softmax(mixed_logits, dim=1) * mixed_target, dim=1))

        # penalty regularization for mixed labeled data
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(mixed_logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # label consistency regularization for unlabeled data
        if (args.dataset == 'cifar100') and ((args.r==0.2) or (args.r==0.5)):
            all_inputs, all_labels, all_masks = torch.cat([inputs_us, inputs_xs], dim=0), torch.cat([labels_u,  labels_x], dim=0), torch.cat([mask_u, mask_x], dim=0)
            all_logits = net(all_inputs)
            Lu = (F.cross_entropy(all_logits, all_labels, reduction='none') * all_masks.float()).mean()

        else:
            logits_us = net(inputs_us)
            Lu = (F.cross_entropy(logits_us, labels_u, reduction='none') * mask_u.float()).mean()

        # overall loss
        loss = Lx + penalty + Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.write("\r")
        log.write(
            "%s: %.1f-%s | Epoch [%3d/%3d], Iter[%3d/%3d]\t Lx: %.4f, Lu: %.4f, Lpen: %.4f"
            % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs - 1, batch_idx + 1, num_iter, Lx.item(), Lu.item(), penalty.item())
        )
        log.flush()


def test(epoch, net1, net2, test_log, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()
    
