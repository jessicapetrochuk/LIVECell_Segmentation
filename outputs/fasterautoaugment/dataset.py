import os
import numpy as np
from PIL import Image
import torch.utils.data
from pycocotools import coco
from torchvision import transforms


class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.root = '/content/drive/MyDrive/School/Harvard/NEUROBIO240/Project/Data/train'
        self.coco = coco.COCO('/content/drive/MyDrive/School/Harvard/NEUROBIO240/Project/Data/livecell_coco_train_subset.json')
        self.ids = list(self.coco.imgs.keys())
        self.resize = transforms.Resize((480, 640))
        # Implement additional initialization logic if needed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        anns = coco.loadAnns(ann_ids)
        mask = np.max(np.stack([coco.annToMask(ann) * ann["category_id"] for ann in anns]), axis=0)
        img_path_name = coco.loadImgs(img_id)[0]['file_name']
        img_label = img_path_name.split('_')[0]
        path = os.path.join(img_label, img_path_name)

        img = Image.open(os.path.join(self.root, path))
        mask = np.asarray(self.resize(torch.from_numpy(mask).unsqueeze(0))).astype(np.float32)
        img = np.asarray(self.resize(img))

        image = np.asarray(img).astype(np.float32)
        image = np.stack((image, image, image), 2)
        #mask = np.expand_dims(mask, 0)

        return image, mask