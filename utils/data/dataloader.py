import torch
import pandas as pd
import utils.data.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def _get_transform(image_size, augment):
    if augment:
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomScale(),
                T.Pad(image_size),
                T.RandomCrop(image_size),
                T.PILToTensor(),
                T.Normalize(),
            ]
        )
    else:
        return T.Compose(
            [
                T.Pad(image_size),
                T.PILToTensor(),
                T.Normalize(),
            ]
        )


class _SemanticSegmentationDataset(Dataset):
    def __init__(self, csv, transform):
        self.dataframe = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        example = self.dataframe.iloc[idx]
        image = Image.open(example['image']).convert('RGB')
        anno = Image.open(example['annotation'])
        image, anno = self.transform(image, anno)
        anno = anno.long().squeeze(0)
        return image, anno


def create_dataloader(csv, batch_size, image_size, augment=False, shuffle=False,
                      seed=None, num_workers=0):
    dataset = _SemanticSegmentationDataset(
        csv,
        transform=_get_transform(image_size, augment)
    )

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            generator=g)
    return dataloader
