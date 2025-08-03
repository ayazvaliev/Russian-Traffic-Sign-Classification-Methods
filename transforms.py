import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

DEFAULT_TRANSFORM_TRAIN = None
DEFAULT_TRANSFORM_VALID = None

def init_transforms():
    global DEFAULT_TRANSFORM_TRAIN, DEFAULT_TRANSFORM_VALID

    common_transforms = [
        A.Resize(*config.NET_SIZE),
        A.CenterCrop(*config.CROP_SIZE),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=config.MEAN, std=config.STD, max_pixel_value=1.0),
        ToTensorV2(),
    ]
    augmentations = [
        A.Affine(translate_percent=0.05, scale=1.05, rotate=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ]

    DEFAULT_TRANSFORM_TRAIN = A.Compose(augmentations + common_transforms)
    DEFAULT_TRANSFORM_VALID = A.Compose(common_transforms)

common_transforms = [
    A.Resize(*config.NET_SIZE),
    A.CenterCrop(*config.CROP_SIZE),
    A.ToFloat(max_value=255.0),
    A.Normalize(mean=config.MEAN, std=config.STD, max_pixel_value=1.0),
    ToTensorV2(),
]

augmentations = [
  A.Affine(translate_percent=0.05, scale=1.05, rotate=15, p=0.5),
  A.HorizontalFlip(p=0.5),
  A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
  A.RandomBrightnessContrast(p=0.5),
]

DEFAULT_TRANSFORM_TRAIN = A.Compose(augmentations + common_transforms)
DEFAULT_TRANSFORM_VALID = A.Compose(common_transforms)