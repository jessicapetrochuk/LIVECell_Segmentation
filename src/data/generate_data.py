import os
import math
import torch
import typing
import numpy as np
import torchio as tio
import albumentations as A
import torchvision.transforms as transforms

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from .bigaug import RandomSharpen, RandomBrightnessShift, RandomIntensityPerturbation

bigaug_transforms = tio.Compose([
        tio.ToCanonical(),
        tio.ZNormalization(), 
        tio.RandomBlur(std=(0.25,1.5), p = 0.5),
        RandomSharpen(std1=(0.25,1.5),std2=(0.25,1.5), alpha=(10,30), p = 0.5), 
        tio.RandomNoise(std=(0.1,1), p = 0.5),
        RandomBrightnessShift(shift_range=(0, 0.1), p = 0.5),
        tio.RandomGamma(log_gamma=(math.log(0.5), math.log(4.5)), p =0.5), 
        RandomIntensityPerturbation(shift_range=(0,0.1),scale_range=(0,0.1), p = 0.5),
        tio.RandomAffine(degrees=(-20,20), p = 0.5),
        tio.RandomAffine(scales=(0.4,1.6), p = 0.5),
        tio.RandomElasticDeformation(num_control_points=(7, 7, 7),locked_borders=2,p= 0.5)
])  


def get_img_paths(directory: str) -> typing.List[str]:
    img_paths = []

    for cell_type in os.listdir(directory):
        path_to_folder = f"{directory}/{cell_type}/"
        if not cell_type.startswith('.'):
            for file in os.listdir(path_to_folder):
                full_path: str = os.path.join(path_to_folder, file)
                img_paths.append(full_path)

    return img_paths
    

class SegmentationDataset(Dataset):
    def __init__(self, root: str, annotations, transform):
        img_paths = get_img_paths(root)

        self.root = root
        self.coco = annotations
        self.ids = list(self.coco.imgs.keys())
        self.transformlabel = transform
        self.resize = transforms.Resize((480, 640))
        self.img_paths = img_paths
        
        if transform == 'bigaug':
            self.transform = bigaug_transforms
        if transform == 'base':
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        if transform == 'autoaugment':
            self.transform = A.load("models/fasterautoaugment/policy/final.json")

    def __getitem__(self, index):
        # Get mask
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = torch.LongTensor(np.max(np.stack([coco.annToMask(ann) * ann["category_id"] 
                                                 for ann in anns]), axis=0)).unsqueeze(0)
        
        # Get image
        img_path_name = coco.loadImgs(img_id)[0]['file_name']
        img_label = img_path_name.split('_')[0]
        path = os.path.join(img_label, img_path_name)
        img = Image.open(os.path.join(self.root, path))

        # Transform mask and image
        if self.transformlabel == 'bigaug':
            img = transforms.PILToTensor()(img)
            img = self.resize(img)
            mask = self.resize(mask)
            
            img = torch.unsqueeze(img, 3)
            mask = torch.unsqueeze(mask, 3)

            subject = tio.Subject(
                image = tio.ScalarImage(tensor=img),
                mask = tio.LabelMap(tensor=mask)
            )

            subject_transform = self.transform(subject)
            img = subject_transform.get_images_dict(intensity_only=False)['image'].data.squeeze(3)
            mask = subject_transform.get_images_dict(intensity_only=False)['mask'].data.squeeze(3)
        
        if self.transformlabel == 'autoaugment':
            mask = np.asarray(self.resize(mask)).astype(np.float32)
            img = np.asarray(self.resize(img))

            image = np.asarray(img).astype(np.float32)
            image = np.stack((image, image, image), 2)
            mask = mask.squeeze(0)

            transformed = self.transform(image=image, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        if self.transformlabel == 'base':
            img = self.resize(img)
            mask = self.resize(mask)
            img = self.transform(img)

        if img.shape[0] == 1:
            img = torch.cat([img]*3)

        return (img, mask)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def get_img_paths(directory: str) -> typing.List[str]:
        return get_img_paths(directory)