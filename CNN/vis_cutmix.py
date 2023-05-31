import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import cv2

transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,shuffle=False, num_workers=4)
alpha = 0.2
cutmix_prob = 0.1
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

for data in trainloader:
    inputs, labels = data
    r = 0.05
    if alpha > 0 and r < cutmix_prob:
            # generate mixed sample
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(inputs.size()[0])
        target_a = labels
        target_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        inputs = inputs.permute(0, 2, 3, 1).numpy() * 255.0

    cv2.imwrite("img/img4.jpg", inputs[0])
    cv2.imwrite("img/img5.jpg", inputs[1])
    cv2.imwrite("img/img6.jpg", inputs[2])
    break