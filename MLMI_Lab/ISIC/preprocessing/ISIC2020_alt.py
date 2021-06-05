import cv2
import os
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, path, image_ids, labels=None, transforms=None):
        super().__init__()
        self.path = path
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.path}/{image_id}.jpg', cv2.IMREAD_COLOR)
        
        if not os.path.exists(f'{self.path}/{image_id}.jpg'):
            print(f'{self.path}/{image_id}.jpg')

        if self.transforms:
            sample = self.transforms(image=image)
            image  = sample['image']

        label = self.labels[idx] if self.labels is not None else 0.5
        return image, label

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)