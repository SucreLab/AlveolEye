import os
import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
import json
import cv2

import automated_lung_morphometry.lungcv.mrcnn.utils as utils
from alveoleye.lungcv.mrcnn.engine import train_one_epoch
from alveoleye.lungcv.model_operations import init_untrained_model, get_transform

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train: bool, n_img_load=None, img_extension=".png", n_repeat_images=1):
        self.root = root
        self.transforms = transforms
        self.train = train
        self._loaded_images = {}

        if train:
            self.folder = "train"
        else:
            self.folder = "val"

        imgs = list(sorted(os.listdir(os.path.join(root, f"images/{self.folder}"))))
        masks = list(sorted(os.listdir(os.path.join(root, f"masks/{self.folder}"))))

        self.imgs = [img for img in imgs if img.endswith(img_extension)] * n_repeat_images
        self.masks = [mask for mask in masks if mask.endswith(img_extension)] * n_repeat_images

        if n_img_load is not None:
            self.imgs = self.imgs[:n_img_load]
            self.masks = self.masks[:n_img_load]

        if not len(self.imgs) == len(self.masks):
            raise ValueError("The number of images and number of masks is not equal, check dataset")

        self.class_dict = self._load_classes(self.root)

    def __getitem__(self, idx):
        if idx in self._loaded_images.keys():

            img = self._loaded_images[idx][0]
            target = self._loaded_images[idx][1]

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        img_path = os.path.join(self.root, f"images/{self.folder}", self.imgs[idx])
        mask_path = os.path.join(self.root, f"masks/{self.folder}", self.masks[idx])

        img = Image.open(img_path).convert("RGB")  # Also need to flip image to match masks?
        img = ImageOps.mirror(img)

        masks, labels = self._rgb_to_class_mask_list(mask_path, self.class_dict)

        num_objs = len(labels)
        boxes = []
        exclude_instanceid = []

        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            # Quick hack to drop boxes that have invalid dimensions
            # This does not address the fact that invalid masks are made in the code - need to look at that
            if (xmax - xmin < 2) or (ymax - ymin < 2):
                exclude_instanceid.append(i)

        boxes = np.array([item for id, item in enumerate(boxes) if id not in exclude_instanceid])
        labels = np.array([item for id, item in enumerate(labels) if id not in exclude_instanceid])
        masks = np.array([item for id, item in enumerate(masks) if id not in exclude_instanceid])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = 0

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        self._loaded_images[idx] = [img, target]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _load_classes(self, root):
        colors = open(os.path.join(root, "classes.json"))
        colors = json.load(colors)

        class_colors = {}

        for number, name in enumerate(colors):
            class_colors[colors[name]] = number + 1
        return class_colors

    def _rgb_to_class_mask_list(self, mask_path, class_colors):

        path = os.path.join(mask_path)

        path = os.path.join(mask_path)
        mask_img = np.array(Image.open(path).convert("RGB"))
        ncol = mask_img.shape[-1]
        if ncol == 4:
            mask_img = mask_img[:, :, 0:3]

        components = []
        num_ids = []

        print(f"[+] Loading {path}")

        for color, num_id in class_colors.items():
            color = color.replace('[', '')
            color = color.replace(']', '')
            color = color.split(' ')
            for number, value in enumerate(color):
                color[number] = int(value)

            mask = cv2.inRange(mask_img, np.clip(np.array(color) - 15, 0, 255), np.clip(np.array(color) + 15, 0, 255))
            mask = np.where(mask == 255, 1, mask)

            number_of_labels, labeled = cv2.connectedComponents(mask)
            blob_mask = [labeled == i for i in range(1, number_of_labels)]

            blob_mask = [blob for blob in blob_mask if np.sum(blob > 0) > 15]
            components.extend(blob_mask)
            num_ids.extend([num_id] * len(blob_mask))

        # Handle the case where the image does not have any annotations
        if len(components) > 0:
            final_mask = np.stack(components, axis=0).astype(np.uint8)
        else:
            final_mask = np.zeros(mask_img.shape)

        return final_mask, num_ids

    def __len__(self):
        return len(self.imgs)


def train_model(num_classes=3):
    dataset = LungDataset('png_dataset', get_transform(train=True), train=True, n_repeat_images=2)
    dataset_val = LungDataset('png_dataset', get_transform(train=False), train=False, n_repeat_images=2)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=10,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = init_untrained_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1
    )

    num_epochs = 1000

    writer = SummaryWriter()
    images, labels = next(iter(data_loader))
    grid = make_grid(list(images))
    writer.add_image('train_images', grid, 0)
    val_images, val_labels = next(iter(data_loader_test))
    val_grid = make_grid(list(val_images))
    writer.add_image('val_images', val_grid, 0)

    for epoch in range(num_epochs):
        # Run training
        model.train(True)
        train_metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        if epoch % 50 == 0 and epoch != 0:
            print(f"Saving intermediate model to: pytorch_trained_model_{epoch}_of_{num_epochs}.pth")
            torch.save(model, f'pytorch_trained_model_{epoch}_of_{num_epochs}.pth')
        lr_scheduler.step()

        # Run validation
        val_metrics = utils.eval_forward(model, data_loader_test, device)[0]

        for metric in train_metrics.meters.keys():
            if isinstance(train_metrics.meters[metric], utils.SmoothedValue):
                value = train_metrics.meters[metric].value
            else:
                value = train_metrics.meters[metric]
            writer.add_scalar(f"{metric}/train", value, epoch)

        for metric in val_metrics.keys():
            if isinstance(val_metrics[metric], utils.SmoothedValue):
                value = val_metrics[metric].value
            else:
                value = val_metrics[metric]
            writer.add_scalar(f"{metric}/val", value, epoch)
        writer.flush()

    print("[+] Training completed")

    torch.save(model, 'pytorch_trained_model.pth')


train_model()
