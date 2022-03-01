import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import skimage.io as io
from model import Modified_VGG16
import matplotlib.pyplot as plt
import time
import random
import cv2

def load_transform_image(path):
    image = io.imread(path)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = (1.0/256) * image
    image = torch.from_numpy(image)
    return image

class SSID_Dataset(Dataset):

    GT_name = "GroundTruth\\"
    NOISY_name = "Noisy\\"

    train_size = 10000

    def __init__(self,root,use_gpu = False):
        self.root = root

        self.data_names = os.listdir(self.root + self.GT_name)

        random.shuffle(self.data_names)

        self.use_gpu = use_gpu

        self.len = self.train_size

    def __getitem__(self, index):
        X = load_transform_image(self.root + self.NOISY_name + self.data_names[index])
        Y = load_transform_image(self.root + self.GT_name + self.data_names[index])
        if self.use_gpu:
            X = X.cuda()
            Y = Y.cuda()
        return X , Y

    def __len__(self):
        return self.len

    def shuffle_data(self):
        random.shuffle(self.data_names)

def train(epoch,loader,model,optimizer,criterion,use_gpu = True):
    cost = 0.0
    for batch_index, (X, Y) in enumerate(loader,0):
        optimizer.zero_grad()
        if use_gpu:
            X = X.cuda()
            Y = Y.cuda()
        prediction = model(X)
        loss = criterion(prediction,Y)
        loss.backward()
        optimizer.step()

        cost += loss.item()
    print("The %d is done! Now the loss is %f." % (epoch,cost))

if __name__ == "__main__":

    use_gpu = True

    alpha = 0.001

    train_dataset = SSID_Dataset("D:\\Dataset\\SIDD_Small_sRGB_Only\\Data\\",use_gpu=use_gpu)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=6,shuffle=False)

    print("数据/缓存初试化完成")

    DUnet = Modified_VGG16()

    print("网络模型初试化完成")

    criterion = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(DUnet.parameters())

    print("优化器初试化完成")

    if use_gpu:
        DUnet = DUnet.cuda()
        criterion = criterion.cuda()

    print("开始训练")

    costs = []

    for epoch in range(10):
        before = time.time()

        cost = train(epoch, train_dataloader, DUnet, optimizer, criterion,use_gpu)

        after = time.time()
        print("训练第%d轮用时:%.2fs" % (epoch,after - before))
        costs.append(cost)

        torch.save(DUnet.state_dict(), 'DUnet-200.pth')
        train_dataset.shuffle_data()
        print("data has been shuffled.")



    plt.plot([_ for _ in range(1, len(costs) + 1)], costs)
    plt.show()