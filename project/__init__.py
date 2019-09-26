import torch
from .models import *
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
    torch.manual_seed(seed)

    path = f"project/weights/best/srcnn_l2_multi_bicubic_100.pth"
    tensor_to_image = transforms.ToPILImage()
    image_to_tensor = transforms.ToTensor()


    model = SRCNN()

    checkpoint = torch.load(path, map_location='cpu')


    model.load_state_dict(checkpoint['model_state_dict'])
    img = image_to_tensor(img)
    model.eval()


    pred = model(img.unsqueeze(0))
    pred = pred.squeeze(0).to(torch.device('cpu'))
    return tensor_to_image(pred.detach())
