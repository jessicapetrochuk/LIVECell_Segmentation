import os
from pycocotools import coco

from ..constants import *
from .generate_data import SegmentationDataset


def load_data():
    train_annotations = coco.COCO('data/livecell_coco_train_subset.json')
    test_annotations = coco.COCO('data/livecell_coco_test_subset.json')

    root = 'data'
    train_test_dict = {
        'train': train_annotations,
        'test': test_annotations
    }

    dsets = {set: SegmentationDataset(os.path.join(root, set), annotations, transform=TRANSFORM)
            for set, annotations in train_test_dict.items()}

    train_dataset = SegmentationDataset(root = 'data/train', annotations = train_annotations, transform=TRANSFORM)
    test_dataset = SegmentationDataset(root = 'data/test', annotations = test_annotations, transform="base")

    return dsets, train_dataset, test_dataset
