import torchvision.models as models
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2);

    def forward(self,x):
        #out = x
        out = nn.functional.interpolate(x, mode='bicubic', scale_factor=4.0)
        #out = self.upsample(x)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out

class SRCNN2(nn.Module):
    def __init__(self):
        super(SRCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)

    def forward(self,x):
        out = nn.functional.interpolate(x, mode='bicubic', scale_factor=4.0)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out

class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        # First Part
        self.conv1 = nn.Conv2d(3,56,kernel_size=5,padding=0)
        self.prelu1 = nn.PReLU()
        # Middle Part Start
        self.conv2 = nn.Conv2d(56,16,kernel_size=1,padding=0)
        self.prelu2 = nn.PReLU()
        # m Part
        self.conv3 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.prelu4 = nn.PReLU()
        self.conv4 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.prelu4 = nn.PReLU()
        self.conv4 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.prelu4 = nn.PReLU()
        # Middle Part End
        self.conv5 = nn.Conv2d(16,56,kernel_size=1,padding=0)
        self.prelu5 = nn.PReLU()
        # Last Part
        self.conv6 = nn.ConvTranspose2d(56,3,kernel_size=9, stride=4, padding=3)

    def forward(self, x):
        out = self.prelu1(self.conv1(x))
        out = self.prelu2(self.conv2(out))
        out = self.prelu3(self.conv3(out))
        out = self.prelu4(self.conv4(out))
        out = self.prelu5(self.conv5(out))
        out = self.conv6(out)
        out = nn.functional.interpolate(out, mode='bicubic', size=(256, 256))
        return out


class loss_net(nn.Module):
    def __init__(self):
        super(loss_net, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.layer_map = {
        "3":"relu1_2",
        "8":"relu2_2",
        "15":"relu3_3",
        "22":"relu4_3"
        }
        for param in self.vgg.parameters():
            param.requires_grad = False

    def normalize(self, batch):
        # normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch.div_(255.0)
        return (batch - mean) / std

    def forward(self, x):
        #x = self.normalize(x)
        out = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_map:
                out[self.layer_map[name]] = x
        return out

class SRResnet(nn.Module):
    def __init__(self):
        super(SRResnet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,64,kernel_size=9, stride=1, padding=4)
        self.b1 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.b2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.b3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.relu(self.b1(self.conv1(x)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.relu(self.b2(self.conv2(y)))
        y = self.relu(self.b3(self.conv3(y)))
        return self.conv4(y)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x
        out = self.relu(self.b1(self.conv1(x)))
        out = self.b2(self.conv2(out))
        out = out + res
        return out

class SRResnet2(nn.Module):
    def __init__(self):
        super(SRResnet2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,64,kernel_size=9, stride=1, padding=4)
        self.b1 = nn.InstanceNorm2d(64,affine=True)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.conv2 = UpsampleConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, upsample=2)
        self.b2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = UpsampleConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, upsample=2)
        self.b3 = nn.InstanceNorm2d(64, affine=True)
        self.conv4 = UpsampleConv(in_channels=64, out_channels=3, kernel_size=9, stride=1)
    def forward(self, x):
        y = self.relu(self.b1(self.conv1(x)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.relu(self.b2(self.conv2(y)))
        y = self.relu(self.b3(self.conv3(y)))
        return self.conv4(y)

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConv, self).__init__()
        self.upsample = upsample
        padding = kernel_size//2
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
    def forward(self, x):
        y = x
        if self.upsample:
            y = nn.functional.interpolate(y, mode='nearest', scale_factor=self.upsample)
        y = self.conv1(y)
        return y
