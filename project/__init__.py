import torch
from .models import SRCNN, SRResnet
    
def superresolve(img, seed=2019):
    """
    YOUR CODE HERE
    
    Superresolve an image given the factor
    @img: A numpy float array
    @return A numpy float array
    
    e.g:
    >>> img.shape
    >>> (64,64,3)
    >>> img = superresolve(img, 4)
    >>> (256,256,3)
    """
    path = f"./srresnet_20epochs.pth"
    #model = SRResnet()
    model = SRCNN()
    model.load_state_dict(torch.load(path))
    pred = model(torch.Tensor(img).unsqueeze(0))
    return pred.squeeze(0).transpose(0, 2).numpy()
    
    
