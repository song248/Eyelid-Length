import os
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_url, check_integrity

from PIL import Image
import albumentations as A
import cv2

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class CustomSegmentation(data.Dataset):
    cmap = voc_cmap()

    def __init__(self,
                 root='datasets',
                 image_set='train',
                 transform=None):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
    
        self.image_dir = os.path.join(self.root, 'images')
        self.mask_dir = os.path.join(self.root, 'masks')
        train_img, val_img = train_test_split(os.listdir(self.image_dir), test_size=0.2, shuffle=True, random_state=42)
        all_img = os.listdir(self.image_dir)
        train_mask = [x.replace('.jpg', '.png') for x in train_img]
        val_mask = [x.replace('.jpg', '.png') for x in val_img]
        test_mask = [x.replace('.jpg', '.png') for x in all_img]

        if image_set == 'train':
            print('train')
            self.images = sorted([os.path.join(self.image_dir,x) for x in train_img])
            self.masks = sorted([os.path.join(self.mask_dir,x) for x in train_mask])
        
        elif image_set == 'val':
            print('val')
            self.images = sorted([os.path.join(self.image_dir,x) for x in val_img])
            self.masks = sorted([os.path.join(self.mask_dir,x) for x in val_mask])
        elif image_set == 'test':
            print('test')
            self.images = sorted([os.path.join(self.image_dir,x) for x in all_img])
            self.masks = sorted([os.path.join(self.mask_dir,x) for x in test_mask])
        
        #self.images = [x for x in os.listdir(self.image_dir)]
        #self.masks = [x for x in os.listdir(self.mask_dir)]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        #img = Image.open(self.images[index]).convert('RGB')
        #target = Image.open(self.masks[index])
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        target = cv2.imread(self.masks[index])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            data = self.transform(image = img, mask=target)
            data_img = data['image'] 
            data_target = data['mask'] 

        return data_img, data_target#img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

if __name__ == '__main__':
    custom = CustomSegmentation()
    custom()
