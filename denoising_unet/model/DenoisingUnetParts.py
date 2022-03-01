import torch


class Double_Conv(torch.nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_channel, out_channel):
        super(Double_Conv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Triple_Conv(torch.nn.Module):
    '''(conv => BN => ReLU) * 3'''
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down_Dconv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down_Dconv, self).__init__()
        self.maxpool_dconv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            Double_Conv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.maxpool_dconv(x)
        return x

class Down_Tconv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down_Tconv, self).__init__()
        self.maxpool_tconv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            Triple_Conv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.maxpool_tconv(x)
        return x

class Up_Dconv(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Up_Dconv, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)
        self.Dconv = Double_Conv(in_channel,out_channel)

    def forward(self,x1,x2):
        x1 = self.deconv(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.Dconv(x)
        return x

class Up_Tconv(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Up_Tconv, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)
        self.Tconv = Triple_Conv(in_channel,out_channel)

    def forward(self,x1,x2):
        x1 = self.deconv(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.Tconv(x)
        return x

