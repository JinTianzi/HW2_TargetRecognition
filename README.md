# HW2_TargetRecognition
## 1. CNN模型
所需环境：pytorch

采用CIFAR-100数据集，数据存储于data文件夹下

使用ResNet-18作为baseline，模型训练：resnet18_baseline.py，模型可视化：vis_baseline.py

cutmix方法，模型训练：resnet18_cutmix.py，模型可视化：vis_cutmix.py

cutout方法，模型训练：resnet18_baseline.py，模型可视化：vis_cutout.py

mixup方法，模型训练：resnet18_mixup.py，模型可视化：vis_mixup.py


## 2. Faster R-CNN模型
所需环境：tensorflow-gpu==2.2.0

数据集：VOC-2007数据集，存放在VOCdevkit中

模型结构：运行summary.py文件可以看到模型参数结构。


训练步骤：首先运行运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt，接着运行train.py文件进行模型训练，训练权值和日志、loss曲线、mAP曲线会生成在logs文件夹中。

测试步骤：将测试图像存放在img文件夹下，运行predict.py文件，输入测试图像相对路径如
```python
img/test1.jpg
```
可以得到一阶段proposal box可视化，和最终的检测结果可视化。


## 3. FCOS模型
所需环境：pytorch

数据集：VOC-2007数据集，存放在datasets文件夹下


训练步骤：运行tools文件夹下的train.py

测试步骤：运行tools文件夹下的test.py
