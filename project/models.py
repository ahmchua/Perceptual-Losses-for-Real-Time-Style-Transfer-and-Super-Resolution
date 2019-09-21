import torchvision.models as models


class loss_net(torch.nn.Module):
    def __init__(self):
        super(loss_net, self).__init__()
        self.vgg = vgg.vgg16(pretrained=True).features
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
