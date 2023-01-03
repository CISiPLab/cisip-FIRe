from torchvision import transforms
from torchvision.transforms import InterpolationMode


def grayscale_to_rgb():
    return transforms.Lambda(lambda x: x.convert('RGB'))


def interpolation(name):
    name = name.lower()

    if name == 'bilinear':
        return InterpolationMode.BILINEAR
    elif name == 'nearest':
        return InterpolationMode.NEAREST
    else:  # bicubic
        return InterpolationMode.BICUBIC


class NCropsTransform:
    """Take N random crops of one image"""

    def __init__(self, *ts):
        new_ts = []
        for t in ts:
            if isinstance(t, list):
                t = transforms.Compose(t)
            new_ts.append(t)

        self.ts = new_ts

    def __call__(self, x):
        out = []
        for t in self.ts:
            out.append(t(x))
        return out

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.ts:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def unnormalize_transform(norm):
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        3: [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]]
    }[norm]

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    return inv_normalize


def normalize_transform(norm):
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        3: [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]]
    }[norm]

    normalize = transforms.Normalize(
        mean=mean,
        std=std
    )

    return normalize


def to_pil():
    return transforms.ToPILImage()
