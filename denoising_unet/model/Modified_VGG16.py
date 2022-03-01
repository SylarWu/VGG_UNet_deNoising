import torch
import torch.nn as nn
from torchvision import models
from .Modified_VGG16_Parts import *

'''
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace=True)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace=True)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace=True)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace=True)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
'''

class Modified_VGG16(nn.Module):
    def __init__(self,in_channel = 3,out_channel = 3):
        super(Modified_VGG16,self).__init__()

        model = models.vgg16(pretrained=True)

        features = model.features

        self.inconv = nn.Sequential(
            features[0],
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            features[2],
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.down_conv1 = nn.Sequential(
            torch.nn.MaxPool2d(2),
            features[5],
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            features[7],
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down_conv2 = nn.Sequential(
            torch.nn.MaxPool2d(2),
            features[10],
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            features[12],
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            features[14],
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down_conv3 = nn.Sequential(
            torch.nn.MaxPool2d(2),
            features[17],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            features[19],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            features[21],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.down_conv4 = nn.Sequential(
            torch.nn.MaxPool2d(2),
            features[24],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            features[26],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            features[28],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        for p in self.parameters():
            p.requires_grad = False

        self.up_conv1 = Up_Tconv(512,512,256)
        self.up_conv2 = Up_Tconv(256,256,128)
        self.up_conv3 = Up_Tconv(128,128,64)
        self.up_conv4 = Up_Dconv(64,64,64)
        self.outconv1 = Double_Conv(64, 64)
        self.outconv2 = nn.Conv2d(64, out_channel, kernel_size=1)


    def forward(self,x):
        # 64
        x1 = self.inconv(x)
        # 128
        x2 = self.down_conv1(x1)
        # 256
        x3 = self.down_conv2(x2)
        # 512
        x4 = self.down_conv3(x3)
        # 512
        x5 = self.down_conv4(x4)

        x = self.up_conv1(x5, x4)
        x = self.up_conv2(x, x3)
        x = self.up_conv3(x, x2)
        x = self.up_conv4(x, x1)
        x = self.outconv1(x)
        x = self.outconv2(x)
        return x



if __name__ == "__main__":

    Unet = Modified_VGG16()
