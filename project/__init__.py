import torch
from .models import SRCNN, SRResnet, SRResnet2
from torchvision import transforms


def superresolve(img, seed=2019):
    """
    YOUR CODE HERE
    
    Superresolve an image by a factor of 4
    @img: A PIL image
    @return A PIL image
    
    e.g:
    >>> img.size
    >>> (64,64)
    >>> img = superresolve(img)
    >>> img.size
    >>> (256,256)
    """
    
    path = f"project/srresnet_20epochs.pth"
    tensor_to_image = transforms.ToPILImage()
    image_to_tensor = transforms.ToTensor()

    model = SRResnet()
    model.eval()
    
    model.load_state_dict(torch.load(path, map_location='cpu'))
    img = image_to_tensor(img)


    pred = model(img.unsqueeze(0))
    pred = pred.squeeze(0).to(torch.device('cpu'))
    return tensor_to_image(pred.detach())
