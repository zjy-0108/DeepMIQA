from torchvision import models
import torch
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.vgg16(pretrained=True)
vgg_feature = model.features

root_dir = "/root/siton-tmp/ct3_jpg"

import os
from PIL import Image
imgs_name = os.listdir(root_dir)

imgs_path = []
labels_data = []

for name in imgs_name:
    if name.endswith("input.jpg"):
        label = 0
    if name.endswith("target.jpg"):
        label = 1
    img_path = os.path.join(root_dir,name)
    imgs_path.append(img_path)
    labels_data.append(label)

from sklearn.model_selection import train_test_split
train_imgs_path,test_imgs_path,train_labels,test_labels = train_test_split(imgs_path,labels_data,test_size=0.25,shuffle=True,stratify=labels_data,random_state=42)

import matplotlib.pyplot as plt
fig = plt.figure()

for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  img = Image.open(train_imgs_path[i])
  plt.imshow(img, interpolation='none')
  plt.title("label: {}".format("ndct" if train_labels[i] == 1 else "ldct"))
  plt.xticks([])
  plt.yticks([])
plt.show()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

my_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


class Dataset(Dataset):
    def __init__(self, imgs_path, labels, my_transforms):
        self.my_transforms = my_transforms
        self.imgs_path = imgs_path
        self.labels = labels
        self.len = len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        return my_transforms(img), self.labels[index]

    def __len__(self):
        return self.len


train_dataset = Dataset(train_imgs_path, train_labels, my_transforms)
test_dataset = Dataset(test_imgs_path, test_labels, test_transforms)

seed=42
import random
import numpy as np
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()
def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12,pin_memory=True, worker_init_fn=_init_fn)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,num_workers=12,pin_memory=True, worker_init_fn=_init_fn)

import torch.nn.functional as F
import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

class MyNet(nn.Module):
    def __init__(self,vgg_feature):
        super(MyNet,self).__init__()
        self.vgg_feature = vgg_feature
        self.ECA = ECA(64)

        self.fc = nn.Sequential(
            nn.Linear(1024*2,256),
            nn.Dropout(0.1),
            nn.Linear(256,2)
        )
    def forward(self,x):
        x = self.vgg_feature(x)
        x = self.ECA(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return F.log_softmax(x,dim=1)
my_net = MyNet(vgg_feature).to(device)

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

log_writer = SummaryWriter(log_dir='/root/tensorboard')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_net.parameters(),lr=0.0005,momentum=0.8)

def train_loss_acc():
    correct = 0
    total = 0
    losses = 0
    for i, data in enumerate(train_loader):
        train_imgs, train_labels = data
        train_imgs = train_imgs.to(device)
        train_labels = train_labels.to(device)
        outputs = my_net(train_imgs)
        _, predict_label = torch.max(outputs, 1)
        total += train_labels.size(0)
        correct += (predict_label == train_labels).sum().item()
        loss = criterion(outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(my_net.parameters(), max_norm=0.4)

        optimizer.step()
        losses += loss.item()

    return losses / (i + 1), correct / total

def test_loss_acc():
    losses = 0
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        test_imgs, test_labels = data
        test_imgs = test_imgs.to(device)
        test_labels = test_labels.to(device)
        outputs = my_net(test_imgs)
        loss = criterion(outputs, test_labels)
        losses += loss.item()
        _, predict_label = torch.max(outputs, 1)
        total += test_labels.size(0)
        correct += (predict_label == test_labels).sum().item()
    return losses / (i + 1), correct / total

best_loss=float('inf')
best_acc=float(0)
epochs_without_improvement = 0
path_model = "zjmodel.pkl"
path_state_dict = "zjmodel_state_dict.pkl"

for epoch in range(0, 200):
    my_net.train()
    train_loss, train_acc = train_loss_acc()

    log_writer.add_scalar("Loss/train", float(train_loss), epoch)
    log_writer.add_scalar("Acc/train", float(train_acc), epoch)

    my_net.eval()
    with torch.no_grad():
        test_loss, test_acc = test_loss_acc()
        if test_loss <= best_loss and test_acc >= best_acc:
            best_loss = test_loss
            best_acc = test_acc
        log_writer.add_scalar("Loss/test", float(test_loss), epoch)
        log_writer.add_scalar("Acc/test", float(test_acc), epoch)

    print("epoch:{} 训练集准确率：{}，loss：{:.6}, 测试集：{}，loss：{:.6}".format(epoch, train_acc, train_loss, test_acc, test_loss))

log_writer.close()

test_loss,test_acc = test_loss_acc()
print("最好的准确率为：{}，最好的损失函数为：{}。".format(best_acc,best_loss))
