import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

resolution = 224 # 456 - consider using higher resolution on Google Cloud
input_res = 512


def get_train_transforms():
    return A.Compose([
        A.ImageCompression(p=0.5),
        A.Rotate(limit=80, p=1.0),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.IAAPiecewiseAffine(),
        ]),
        A.RandomSizedCrop(min_max_height=(int(resolution * 0.7), input_res),
                          height=resolution, width=resolution, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
        ]),
        A.Cutout(num_holes=8, max_h_size=resolution // 8, max_w_size=resolution // 8, fill_value=0, p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ], p=1.0)


def get_valid_transforms():
    return A.Compose([
        A.CenterCrop(height=resolution, width=resolution, p=1.0),
        A.Normalize(),
        ToTensorV2(),
    ], p=1.0)


def get_tta_transforms():
    return A.Compose([
        A.ImageCompression(p=0.5),
        A.RandomSizedCrop(min_max_height=(int(resolution * 0.9), int(resolution * 1.1)),
                          height=resolution, width=resolution, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ], p=1.0)
