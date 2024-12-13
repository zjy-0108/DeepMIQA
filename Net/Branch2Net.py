import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings("ignore")

data=pd.read_csv('F10.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
X=torch.FloatTensor(X)
y=torch.LongTensor(y)

batch = 512
no_of_batches = len(data)//batch
epochs = 100

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_ds = TensorDataset(X_train,y_train)
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True,pin_memory=True, worker_init_fn=_init_fn)
test_ds = TensorDataset(X_test,y_test)
test_dl = DataLoader(test_ds,batch_size=batch,pin_memory=True, worker_init_fn=_init_fn)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(10,32)
        self.liner_2 = nn.Linear(32, 32)
        self.liner_3 = nn.Linear(32,2)
    def forward(self,input):
        x = F.relu(self.liner_1(input))
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x)
        return x

model = Model()
loss_fn = nn.CrossEntropyLoss()

def accuracy(y_pred,y_true):
    y_pred = torch.argmax(y_pred,dim=1)
    acc = (y_pred == y_true).float().mean()
    return acc

train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]

from torch.optim import lr_scheduler

def get_model():
    model = Model()
    opt = torch.optim.Adam(model.parameters(),lr=0.0001)
    return model,opt

model,optim = get_model()
best_loss=float('inf')
best_acc=float(0)
path_model = "model.pkl"
path_state_dict = "model_state_dict.pkl"

from torch.utils.tensorboard import SummaryWriter
log_writer = SummaryWriter(log_dir='/root/tensorboard')

for epoch in range(epochs):
    for x,y in train_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    epoch_accuracy = accuracy(model(X_train),y_train)
    epoch_loss = loss_fn(model(X_train), y_train).data
    with torch.no_grad():
        epoch_test_accuracy = accuracy(model(X_test),y_test)
        epoch_test_loss = loss_fn(model(X_test), y_test).data
        print('epoch: ',epoch,'train_loss: ',round(epoch_loss.item(),4),'train_accuracy: ',round(epoch_accuracy.item(),4),
             'test_loss: ',round(epoch_test_loss.item(),4),'test_accuracy: ',round(epoch_test_accuracy.item(),4)
              )
        if epoch_test_loss.item() <= best_loss and epoch_test_accuracy.item() >= best_acc:
            best_loss = epoch_test_loss.item()
            best_acc = epoch_test_accuracy.item()
            torch.save(model, path_model)
            net_state_dict = model.state_dict()
            torch.save(net_state_dict, path_state_dict)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_accuracy)
        log_writer.add_scalar("Loss/train", float(epoch_loss.item()), epoch)
        log_writer.add_scalar("Acc/train", float(epoch_accuracy.item()), epoch)
        log_writer.add_scalar("Loss/test", float(epoch_test_loss.item()), epoch)
        log_writer.add_scalar("Acc/test", float(epoch_test_accuracy.item()), epoch)

log_writer.close()

print("最好的准确率为：{}，最好的损失函数为：{}。".format(best_acc,best_loss))