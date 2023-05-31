import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import cv2

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,shuffle=False, num_workers=4)

for data in trainloader:
    inputs, labels = data
    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, 1.0)

    inputs = inputs.permute(0, 2, 3, 1).numpy() * 255.0
    
    cv2.imwrite("img/img10.jpg", inputs[0])
    cv2.imwrite("img/img11.jpg", inputs[1])
    cv2.imwrite("img/img12.jpg", inputs[2])
    break