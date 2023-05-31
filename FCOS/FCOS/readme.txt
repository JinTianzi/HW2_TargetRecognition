环境配置

该模型是在MMDetection的框架下运行的
首先需要下载并安装Miniconda
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
再用该语句创造并激活conda环境
模型需要使用到pytoech
cpu或者gpu平台都可以，如果用的是gpu平台则用以下语音进行安装
conda install pytorch torchvision -c pytorch
接下来则是安装MMEngine和MMcv，期中需要注意的是MMCV的版本需要大于2.0.0
最后则是配置MMDetection
可以选用以下语句
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

由于github空间限制，需要以vocdevikt/voc2007的路径将voc2007数据集放在datasets文件夹中

在tools文件夹中的train.py和test.py两个文件就是分别用来训练和测试的模型。
