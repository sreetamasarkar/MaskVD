"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import json
import datasets.transforms as T
import torchvision.transforms as Transforms
import numpy as np

# Original Categories
CLASSES = [ 
    'Pedestrian',
    'Car', 
    'Cyclist', 
    'Van', 
    'Truck',  
    'Person_sitting',
    'Tram', 
    'Misc', 
    'DontCare'
    ]
      
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, n_classes, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(n_classes, return_masks)
        # self.class_names = ['Pedestrian', 'Car', 'Cyclist']
        # self.cat_ids = {1:1, 2:2, 3:3, 4:2, 5:2, 6:1, 7:0, 8:0, 9:0}
        # self.cat_ids = {1:3, 2:6, 3:3, 4:6, 5:5, 6:3, 7:25, 8:-1, 9:-1}
        # self.cat_ids = {1:0, 2:1, 3:2, 4:1, 5:1, 6:0, 7:3, 8:3, 9:3}
        self.frame_ids = self.get_frame_ids(ann_file)
        self.n_classes = n_classes
        assert len(self.frame_ids) == len(self.ids), 'frame_ids and ids should have same length'

    def get_frame_ids(self, ann_file):
        f = open(ann_file, 'r')
        anno = json.load(f)
        frame_ids = [anno_img['frame_id'] for anno_img in anno['images']]
        return frame_ids
    
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        frame_id = self.frame_ids[idx]
        target = {'image_id': image_id, 'frame_id': frame_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        annotation = {'boxes': 0, 'labels': 0, 'frame_id': frame_id}
        for key, value in target.items():
            #print(key)
            #print(value)
            if key in ['boxes', 'labels']:
                annotation.update({key: value})
        return img, annotation


class CocoItem(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img,
        target,
        length,
        transforms = None,
        
    ):
        self.img = img
        self.target = target
        self.length = length
        self._transforms = transforms

    def __getitem__(self, index):
        I = self.img.numpy()
        frame = torch.from_numpy((I * 255).astype(np.uint8))
        #print(frame.dtype)
        #frame.dtype = torch.uint8
        annotation = {'boxes': 0, 'labels': 0}
        for key, value in self.target.items():
            #print(key)
            #print(value)
            if key in ['boxes', 'labels']:
                annotation.update({key: value})
        # print("#########Coco Item########")
        # print(frame)
        # print(frame.size())
        # print(frame.shape)
        # print(annotation)
        # print("#########kitt########")
        return frame, annotation

    def __len__(self):
        return self.length




def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, n_classes, return_masks=False):
        self.return_masks = return_masks
        self.n_classes = n_classes

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        frame_id = target["frame_id"]
        image_id = torch.tensor([image_id])
        frame_id = torch.tensor([frame_id])

        anno = target["annotations"]

        # anno = [obj for obj in anno if ('iscrowd' not in obj or obj['iscrowd'] == 0) and obj["category_id"] < self.n_classes]
        anno = [obj for obj in anno if ('iscrowd' not in obj or obj['iscrowd'] == 0) and obj["category_id"] != 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        track_ids = [obj["track_id"] for obj in anno]
        track_ids = torch.tensor(track_ids, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]) & (classes != 0)
        boxes = boxes[keep]
        classes = classes[keep]
        track_ids = track_ids[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["track_ids"] = track_ids
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["frame_id"] = frame_id
        # check if first frame of a sequence
        # target["is_first_frame"] = (frame_id == 1)
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, default_res=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # For frame skipping: resize images to fixed size since pixel threshold is a parameter and has a fixed shape 
    # default res: (w, h)
    if default_res is not None:
        return T.Compose([
            T.RandomResize([default_res]),
            T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            T.ToUint8()
        ])
    
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, n_classes=7, args=None):
    root = Path('/data1/kitti_tracking')
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / 'data_tracking_image_2/training/image_02', root / "annotations_old" / f'tracking_train_half.json'),
        "val": (root / 'data_tracking_image_2/training/image_02', root / "annotations_old" / f'tracking_val_half.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, n_classes=n_classes, transforms=make_coco_transforms(image_set, (672, 370)), return_masks=False)
    return dataset