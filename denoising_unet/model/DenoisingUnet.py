import torch
from .DenoisingUnetParts import *

class DenoisingUnet(torch.nn.Module):
    def __init__(self,in_channel = 3,out_channel = 3):
        super(DenoisingUnet, self).__init__()
        self.inconv = Double_Conv(in_channel,64)
        self.down_conv1 = Down_Dconv(64,128)
        self.down_conv2 = Down_Tconv(128,256)
        self.down_conv3 = Down_Tconv(256,512)
        self.down_conv4 = Down_Tconv(512,1024)
        self.up_conv1 = Up_Tconv(1024,512)
        self.up_conv2 = Up_Tconv(512, 256)
        self.up_conv3 = Up_Tconv(256, 128)
        self.up_conv4 = Up_Tconv(128, 64)
        self.outconv = Double_Conv(64,out_channel)

    def forward(self,x):
        x1 = self.inconv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.down_conv4(x4)
        x  = self.up_conv1(x5,x4)
        x  = self.up_conv2(x,x3)
        x  = self.up_conv3(x,x2)
        x  = self.up_conv4(x,x1)
        x  = self.outconv(x)
        return x
