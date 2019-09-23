#from torchvision.vision import VisionDataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
from PIL import Image
import os
import os.path

def upsample(img, factor):
    w, h = img.size
    return img.resize((int(w*factor), int(h*factor)))

def downsample(img, factor=4.0):
    return upsample(img, 1./factor)

class MyCoco(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, input_transform=None, target_transform=None, transforms=None):
        super(MyCoco, self).__init__()
        from pycocotools.coco import COCO
        self.root = root
        self.transforms = transforms
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_to_tensor = ToTensor()

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        target = img.copy()

        if self.input_transform:
            input = self.input_transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return input, target


    def __len__(self):
        return len(self.ids)
