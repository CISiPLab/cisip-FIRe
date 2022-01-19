import cv2
import kornia
import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


def get_transform_from_config(commands):
    ts = []
    for command in commands:
        ts.append(eval(command))
    ts = transforms.Compose(ts)
    return ts


def get_train_transform(dataset_name, resize, crop, use_rand_aug=False):
    t = {
        'imagenet100': [
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(resize),
            # transforms.RandomCrop(crop),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.05, contrast=0.05),
        ],
        'nuswide': [
            transforms.Resize(resize),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip()
        ],
        'coco': [
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip()
        ],
        'cars': [
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip()
        ],
        'landmark': [
            transforms.RandomResizedCrop(crop),  # follow delg paper
        ],
        'roxford5k': [
        ],
        'rparis6k': [
        ],
        'mirflickr': [
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip()
        ],
        'sop': [
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip()
        ],
        'sop_instance': [
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip()
        ],
        'food101': [
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip()
        ],
    }[dataset_name]

    if use_rand_aug:
        t.insert(0, transforms.RandAugment())
    return t


def get_mean_transforms_gpu(norm=2):
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }[norm]

    return K.Normalize(torch.tensor(mean), torch.tensor(std))


def get_transforms_gpu(dataset, transform_mode='train', resize=0, crop=0) -> nn.Sequential:
    if transform_mode == 'train':
        transform = compose_transform_gpu('train', 0, crop, 2, {
            'imagenet100': [
                K.RandomResizedCrop((crop, crop)),
                K.RandomHorizontalFlip()
            ],
            'nuswide': [
                kornia.Resize((resize, resize)),
                K.RandomCrop((crop, crop)),
                K.RandomHorizontalFlip()
            ],
            'coco': [
                K.RandomResizedCrop((crop, crop)),
                K.RandomHorizontalFlip()
            ],
            'cars': [
                K.RandomResizedCrop((crop, crop)),
                K.RandomHorizontalFlip()
            ],
            'landmark': [
                K.RandomResizedCrop((crop, crop)),  # follow delg paper
            ],
            'landmark200': [
                K.RandomResizedCrop((crop, crop)),  # follow delg paper
            ],
            'landmark500': [
                K.RandomResizedCrop((crop, crop)),  # follow delg paper
            ],
        }[dataset])
    else:
        transform = compose_transform_gpu('test', resize, crop, 2)

    return transform


def compose_transform_gpu(mode='train', resize=0, crop=0, norm=0,
                          augmentations=None):
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }[norm]
    compose = []

    if resize != 0:
        compose.append(kornia.Resize((resize, resize)))

    if mode == 'train' and augmentations is not None:
        compose += augmentations

    if mode == 'test' and crop != 0 and resize != crop:
        compose.append(K.CenterCrop((crop, crop)))

    if norm != 0:
        compose.append(K.Normalize(torch.tensor(mean), torch.tensor(std)))

    return nn.Sequential(*compose)


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class GaussianBlurOpenCV(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class GaussianBlurKornia(object):
    """
    Gaussian Blur using Kornia
    Note that this support gpu operation but required torch.Tensor instead of PIL.Image.

    Here are some performance comparison on cpu measured in ms per image.
    ----------------------------
    |Image size |Kornia |OpenCV |
    -----------------------------
    |224        |10.2   |11.4   |
    |512        |17.1   |16.1   |
    |1024       |41.2   |30.1   |
    |2048       |179    |92.5   |
    -----------------------------

    Examples:
        >>> transform = transforms.Compose([transforms.Resize(224), \
                                            transforms.ToTensor(), \
                                            GaussianBlurKornia(kernel_size=3),])
        >>> transform(img)
    """
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample: torch.Tensor):
        sample = sample.unsqueeze(0)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = kornia.filters.gaussian_blur2d(sample,
                                                    (self.kernel_size, self.kernel_size),
                                                    (sigma, sigma))

        return sample
