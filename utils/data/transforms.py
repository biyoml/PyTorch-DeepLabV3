import torch
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from utils.constants import IMAGE_MEAN, IMAGE_STDDEV, VOID_LABEL


class Compose(object):
    def __init__(self, transforms):
        self.ts = transforms

    def __call__(self, *args):
        for t in self.ts:
            args = t(*args)
        return args


class RandomHorizontalFlip(object):
    def __call__(self, image, anno):
        if random.random() < 0.5:
            image = TF.hflip(image)
            anno = TF.hflip(anno)
        return image, anno


class RandomScale(object):
    def __init__(self, min_scale=0.5, max_scale=2.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, anno):
        w, h = TF.get_image_size(image)
        s = random.uniform(self.min_scale, self.max_scale)
        size = round(min(w, h) * s)
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        anno = TF.resize(anno, size, interpolation=InterpolationMode.NEAREST)
        return image, anno


class Pad(object):
    def __init__(self, image_size):
        self.size = image_size

    def __call__(self, image, anno):
        w, h = TF.get_image_size(image)
        dx = max(self.size - w, 0)
        dy = max(self.size - h, 0)
        image = TF.pad(image, padding=[0, 0, dx, dy], fill=IMAGE_MEAN)
        anno = TF.pad(anno, padding=[0, 0, dx, dy], fill=VOID_LABEL)
        return image, anno


class RandomCrop(object):
    def __init__(self, crop_size):
        self.size = crop_size

    def __call__(self, image, anno):
        w, h = TF.get_image_size(image)
        x1 = random.randint(0, w - self.size)
        y1 = random.randint(0, h - self.size)
        image = TF.crop(image, y1, x1, self.size, self.size)
        anno = TF.crop(anno, y1, x1, self.size, self.size)
        return image, anno


class PILToTensor(object):
    def __call__(self, image, anno):
        image = TF.pil_to_tensor(image)
        anno = TF.pil_to_tensor(anno)
        return image, anno


class Normalize(object):
    def __init__(self):
        self.mean = torch.FloatTensor(IMAGE_MEAN).reshape([-1, 1, 1])
        self.stddev = torch.FloatTensor(IMAGE_STDDEV).reshape([-1, 1, 1])

    def __call__(self, image, anno):
        image = (image - self.mean) / self.stddev
        return image, anno
