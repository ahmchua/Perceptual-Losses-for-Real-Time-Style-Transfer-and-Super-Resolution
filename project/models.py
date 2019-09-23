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
        self.upsample = nn.UpsamplingBilinear2d((256, 256))

    def forward(self,x):
        #out = x
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

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

    def forward(self, x):
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
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0)

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
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.b1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.b2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x
        out = self.relu(self.b1(self.conv1(x)))
        out = self.b2(self.conv2(out))
        out = out + res
        return out
