import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision.transforms.functional as tf
import random
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from classification.utils import tools


# fix randomness on DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def is_image(src):
    return True if os.path.splitext(src)[1].lower() in ['.jpg', '.png', '.tif', '.ppm'] else False


class ImageClassificationLoader_resize(Dataset):
    def __init__(self,
                 root_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        print(f'Loading dataset from {root_path}')
        # Lấy list classes (subfolders)
        self.classes = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        assert len(self.classes) == self.args.n_classes, f'Expected {self.args.n_classes} classes, found {len(self.classes)}'

        self.img_paths = []
        self.labels = []

        # Thu thập images từ mỗi class
        for cls in self.classes:
            cls_path = os.path.join(root_path, cls)
            img_names = sorted(filter(is_image, os.listdir(cls_path)))
            for img_name in img_names:
                self.img_paths.append(os.path.join(cls_path, img_name))
                self.labels.append(self.class_to_idx[cls])

        print(f'{tools.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{tools.Colors.END}')
        self.images = []
        for path in self.img_paths:
            self.images.append(Image.open(path).convert('RGB'))

    def transform(self, image):
        resize_h = self.args.input_size[0]
        resize_w = self.args.input_size[1]
        image = tf.resize(image, [resize_h, resize_w])

        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random

            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)

            if (random_gen.random() < 0.5) and self.args.transform_vflip:
                image = tf.vflip(image)

            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)
            
            if random_gen.random() < 0.5 and getattr(self.args, 'transform_rotation', False):
                transform = transforms.RandomRotation(degrees=self.args.rotation_degrees)
                image = transform(image)
            
            if random_gen.random() < 0.5 and getattr(self.args, 'transform_clahe', False):
                image = tools.clahe_equalized(image)

            if (random_gen.random() < 0.5) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.1)
                image = transform(image)

            # recommend to use at the end.
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)

        image_tensor = tf.to_tensor(image)

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))
            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        return image_tensor

    def __getitem__(self, index):
        img_tr = self.transform(self.images[index])
        label_tr = torch.tensor(self.labels[index])

        return (img_tr, self.img_paths[index]), label_tr

    def __len__(self):
        return len(self.img_paths)


class ImageClassificationLoader_zero_pad(Dataset):
    def __init__(self,
                 root_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        print(f'Loading dataset from {root_path}')
        # Lấy list classes (subfolders)
        self.classes = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        assert len(self.classes) == self.args.n_classes, f'Expected {self.args.n_classes} classes, found {len(self.classes)}'

        self.img_paths = []
        self.labels = []

        # Thu thập images từ mỗi class
        for cls in self.classes:
            cls_path = os.path.join(root_path, cls)
            img_names = sorted(filter(is_image, os.listdir(cls_path)))
            for img_name in img_names:
                self.img_paths.append(os.path.join(cls_path, img_name))
                self.labels.append(self.class_to_idx[cls])

        print(f'{tools.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{tools.Colors.END}')
        self.images = []
        for path in self.img_paths:
            self.images.append(Image.open(path).convert('RGB'))

    def transform(self, image):
        if self.mode == 'validation':
            image = tools.center_padding(image, [int(self.args.input_size[0]), int(self.args.input_size[1])])

        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random

            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)

            if (random_gen.random() < 0.5) and self.args.transform_vflip:
                image = tf.vflip(image)

            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)
            
            if random_gen.random() < 0.5 and getattr(self.args, 'transform_rotation', False):
                transform = transforms.RandomRotation(degrees=self.args.rotation_degrees)
                image = transform(image)
            
            if random_gen.random() < 0.5 and getattr(self.args, 'transform_clahe', False):
                image = tools.clahe_equalized(image)

            # recommend to use at the end.
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)
            
            image = tf.resize(image, [self.args.input_size[0], self.args.input_size[1]]) #Resize to fixed size after all augmentations

        image_tensor = tf.to_tensor(image)

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))
            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        return image_tensor

    def __getitem__(self, index):
        img_tr = self.transform(self.images[index])
        label_tr = torch.tensor(self.labels[index])

        return (img_tr, self.img_paths[index]), label_tr

    def __len__(self):
        return len(self.img_paths)


class ImageClassificationDataLoader_resize:

    def __init__(self,
                 root_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)

        print(f"Creating ImageClassificationDataLoader_resize with root_path: {root_path}, mode: {mode}")

        self.image_loader = ImageClassificationLoader_resize(root_path,
                                                             mode=mode,
                                                             **kwargs)

        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()
    
    def __iter__(self):
        return iter(self.Loader)


class ImageClassificationDataLoader_zero_pad:

    def __init__(self,
                 root_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)

        print(f"Creating ImageClassificationDataLoader_zero_pad with root_path: {root_path}, mode: {mode}")

        self.image_loader = ImageClassificationLoader_zero_pad(root_path,
                                                               mode=mode,
                                                               **kwargs)

        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()
    
    def __iter__(self):
        return iter(self.Loader)


def main():
    # Load YAML config
    config_path = r'./configs/train.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    from types import SimpleNamespace
    args = SimpleNamespace(**config)
    
    # Override for testing (optional)
    args.input_size = [1200, 1200]
    args.n_classes = 3
    args.input_space = 'RGB'
    args.transform_hflip = True
    args.transform_vflip = True
    args.transform_jitter = True
    args.transform_blur = True
    args.transform_perspective = True
    args.transform_rotation = True
    args.rotation_degrees = 30
    args.image_mean = [0.485, 0.456, 0.406]
    args.image_std = [0.229, 0.224, 0.225]

    # Dataset paths
    train_root_path = './dataset/FaRFUM-RoP/'
    val_root_path = './dataset/FaRFUM-RoP/'

    # Initialize train loader
    print(f"Loading train dataset from {train_root_path}")
    train_loader = ImageClassificationDataLoader_resize(
        root_path=train_root_path,
        mode='train',
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        args=args
    )
    
    if len(train_loader) == 0:
        raise ValueError("Train dataset is empty!")
    
    print(f"Train dataset length: {len(train_loader.image_loader)}")
    print(f"Classes: {train_loader.image_loader.classes}")

    # Get first batch
    (inputs, paths), labels = next(iter(train_loader))
    print(f"Input tensor shape: {inputs.shape}")
    print(f"Labels: {labels.tolist()}")
    print(f"Sample paths: {paths[:2]}")

    # Initialize validation loader
    print(f"\nLoading validation dataset from {val_root_path}")
    val_loader = ImageClassificationDataLoader_resize(
        root_path=val_root_path,
        mode='validation',
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        args=args
    )
    
    if len(val_loader) == 0:
        raise ValueError("Validation dataset is empty!")
    
    print(f"Validation dataset length: {len(val_loader.image_loader)}")
    print(f"Validation classes: {val_loader.image_loader.classes}")

    # Get first validation batch
    (val_inputs, val_paths), val_labels = next(iter(val_loader))
    print(f"Val input tensor shape: {val_inputs.shape}")
    print(f"Val labels: {val_labels.tolist()}")

    # Visualize a batch
    def visualize_batch(images, labels, title, input_space, image_mean, image_std):
        if input_space == 'RGB':
            images = images * torch.tensor(image_std).view(1, 3, 1, 1) + torch.tensor(image_mean).view(1, 3, 1, 1)
        images = torch.clamp(images, 0, 1)
        grid = make_grid(images, nrow=2, padding=2)
        grid = grid.permute(1, 2, 0).numpy()
        
        label_strs = [f"Class: {lbl.item()}" for lbl in labels]
        
        plt.figure(figsize=(10, 5))
        plt.imshow(grid)
        plt.title(f"{title}\nLabels: {', '.join(label_strs)}")
        plt.axis('off')
        plt.show()

    visualize_batch(inputs, labels, "Train Batch (Augmented)", args.input_space, args.image_mean, args.image_std)
    visualize_batch(val_inputs, val_labels, "Validation Batch", args.input_space, args.image_mean, args.image_std)

if __name__ == "__main__":
    main()