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

for data in trainloader:
    inputs, labels = data
    inputs = inputs.permute(0, 2, 3, 1).numpy() * 255.0
    
    cv2.imwrite("img/img1.jpg", inputs[0])
    cv2.imwrite("img/img2.jpg", inputs[1])
    cv2.imwrite("img/img3.jpg", inputs[2])
    break