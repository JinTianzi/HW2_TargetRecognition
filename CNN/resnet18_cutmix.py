import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms



transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=4)

t = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=t)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=4)



class ResNet(nn.Module):
    def __init__(self,BasicBlock):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,stride=1,padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=0.2)
        self.BasicBlock1 = BasicBlock(32,32,1)
        self.BasicBlock2 = BasicBlock(32,32,1)
        self.BasicBlock3 = BasicBlock(32,64,2)
        self.BasicBlock4 = BasicBlock(64,64,1)
        self.BasicBlock5 = BasicBlock(64,64,1)
        self.BasicBlock6 = BasicBlock(64,64,1)
        self.BasicBlock8 = BasicBlock(64,128,2)
        self.BasicBlock9 = BasicBlock(128,128,1)
        self.BasicBlock10 = BasicBlock(128,128,1)
        self.BasicBlock11 = BasicBlock(128,128,1)
        self.BasicBlock12 = BasicBlock(128,256,2)
        self.BasicBlock13 = BasicBlock(256,256,1)
        self.fc1 = nn.Linear(1024,100)
        
    def forward(self,x):
        x = self.dropout(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.BasicBlock1(x)
        x = self.BasicBlock2(x)
        x = self.BasicBlock3(x)
        x = self.BasicBlock4(x)
        x = self.BasicBlock5(x)
        x = self.BasicBlock6(x)
        x = self.BasicBlock8(x)
        x = self.BasicBlock9(x)
        x = self.BasicBlock10(x)
        x = self.BasicBlock11(x)
        x = self.BasicBlock12(x)
        x = F.max_pool2d(self.BasicBlock13(x),3,stride=2,padding=1)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        
        return x
        
class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,s):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,out_planes,3,stride=s,padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes,out_planes,3,stride=s,padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_planes)
        self.downsample = nn.Sequential()
        if (s!=1):
            self.downsample = nn.Sequential(nn.Conv2d(in_planes,out_planes,1,stride=2),nn.BatchNorm2d(out_planes))
        
    def forward(self,x):
        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_bn(self.conv2(out))
        x = self.downsample(x)
        out = F.interpolate(out,[x.shape[2],x.shape[3]])
        out += x
        return out
            

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

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
model = ResNet(BasicBlock)
model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 70
alpha = 0.2
cutmix_prob = 0.1

for epoch in range(num_epochs):
    
    total_right = 0
    total = 0
    
    for data in trainloader:
        inputs, labels = data
        r = np.random.rand(1)
        inputs, labels = Variable(inputs).cuda(),Variable(labels).cuda()
        
        optimizer.zero_grad()
        
        if alpha > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(inputs.size()[0])
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            loss = mixup_criterion(loss_fn, outputs, target_a, target_b, lam)
        else:
            # compute output
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)


        loss.backward()
        optimizer.step()
        
        predicted = outputs.data.max(1)[1]
        total += labels.size(0)

        total_right += predicted.eq(labels.data).cpu().sum().float()

    print("Training accuracy for epoch {} : {}".format(epoch+1,total_right/total))
    
    if (epoch+1)%5==0:
        torch.save(model,'hw4_para_cutmix.ckpt')
        
    if (epoch+1)%10==0:
        my_model = torch.load('hw4_para_cutmix.ckpt')

        total_right = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images,labels = data
                images, labels = Variable(images).cuda(),Variable(labels).cuda()
                outputs = my_model(images)
        
                predicted = outputs.data.max(1)[1]
                total += labels.size(0)
                total_right += (predicted == labels.data).float().sum()
        
        print("Test accuracy: %d" % (100*total_right/total))
