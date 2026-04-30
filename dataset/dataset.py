import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from preprocessing.slice_sampling import uniform_slice_sampling
from preprocessing.augmentation import random_augmentation

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class MRData(data.Dataset):
    def __init__(
        self,
        task='acl',
        train=True,
        split=None,
        transform=None,
        weights=None,
        target_slices=32,
        input_dim=INPUT_DIM,
        data_root='./data',
        label_root='./labels',
    ):
        super().__init__()
        self.planes = ['axial', 'coronal', 'sagittal']
        self.records = None
        self.image_path = {}
        self.target_slices = target_slices
        self.input_dim = input_dim
        if split is None:
            split = 'train' if train else 'valid'
        split = split.lower()
        if split not in {'train', 'valid', 'test'}:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self.train = split == 'train'
        self.data_root = data_root
        self.label_root = label_root

        if not self.train:
            transform = None

        self.records = pd.read_csv(
            os.path.join(self.label_root, f'{self.split}-{task}.csv'),
            header=None,
            names=['id', 'label']
        )
        for plane in self.planes:
            self.image_path[plane] = os.path.join(self.data_root, self.split, plane)

        self.transform = transform
        self.records['id'] = self.records['id'].map(lambda i: '0' * (4 - len(str(i))) + str(i))
        
        self.paths = {}
        for plane in self.planes:
            self.paths[plane] = [
                os.path.join(self.image_path[plane], filename + '.npy')
                for filename in self.records['id'].tolist()
            ]

        self.labels = self.records['label'].tolist()
        
        # T??nh to??n weight
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        if weights:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.FloatTensor([neg / pos])
        
        print(f'Task: {task} | Split: {self.split}')
        print(f'Samples: -ve: {neg}, +ve: {pos} | Loss Weights: {self.weights}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        img_raw = {}
        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])
            if self.target_slices is not None:
                img_raw[plane] = uniform_slice_sampling(img_raw[plane], self.target_slices)
            if self.train:
                vol = torch.from_numpy(img_raw[plane])
                vol = random_augmentation(vol)
                img_raw[plane] = vol.numpy()
            img_raw[plane] = self._resize_image(img_raw[plane])
            
        label = self.labels[index]
        label = torch.FloatTensor([1]) if label == 1 else torch.FloatTensor([0])

        return [img_raw[plane] for plane in self.planes], label

    def _resize_image(self, image):
        # 1. Resize/Crop (C???t gi???a ???nh)
        target = self.input_dim
        if target is not None and target <= image.shape[1] and target <= image.shape[2]:
            pad = int((image.shape[2] - target) / 2)
            image = image[:, pad:-pad, pad:-pad]
        
        # 2. Normalize (Chu???n h??a)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * MAX_PIXEL_VAL
        image = (image - MEAN) / STDDEV

        # 3. Chuy???n sang Tensor
        image = torch.FloatTensor(image)

        # 4. QUAN TR???NG: T???o 3 k??nh m??u (RGB)
        # Input ??ang l?? (Slices, H, W) -> Stack th??nh (Slices, 3, H, W)
        image = torch.stack((image,)*3, axis=1)

        # 5. Apply Transform (N???u c??)
        if self.transform:
            # L??c n??y image c?? d???ng (Slices, 3, H, W)
            # torchvision s??? coi 'Slices' l?? batch v?? ??p d???ng transform l??n t???ng slice
            image = self.transform(image)

        return image

def load_data(
    task: str,
    batch_size: int = 1,
    num_workers: int = 0,
    target_slices: int = 32,
    image_size: int = INPUT_DIM,
    data_root: str = './data',
    label_root: str = './labels',
    include_test: bool = False,
):
    # ?????nh ngh??a Augmentation
    # L??u ??: Kh??ng c???n b?????c repeat/permute n???a v?? ???? l??m trong _resize_image
    augments = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)),
        transforms.RandomHorizontalFlip(),
    ])

    print('Loading Train Dataset of {} task...'.format(task))
    train_data = MRData(
        task,
        train=True,
        split='train',
        transform=augments,
        target_slices=target_slices,
        input_dim=image_size,
        data_root=data_root,
        label_root=label_root,
    )
    # Weighted sampling to balance classes in each batch
    labels = train_data.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # num_workers=0 ????? tr??nh l???i tr??n Windows
    train_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=sampler)

    print('Loading Validation Dataset of {} task...'.format(task))
    val_data = MRData(
        task,
        train=False,
        split='valid',
        target_slices=target_slices,
        input_dim=image_size,
        data_root=data_root,
        label_root=label_root,
    )
    val_loader = data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if not include_test:
        return train_loader, val_loader, train_data.weights, val_data.weights

    test_label_path = os.path.join(label_root, f'test-{task}.csv')
    test_data_path = os.path.join(data_root, 'test')
    if not (os.path.exists(test_label_path) and os.path.isdir(test_data_path)):
        print(f"Warning: test split not found for task={task}. Skip loading test set.")
        return train_loader, val_loader, None, train_data.weights, val_data.weights, None

    print('Loading Test Dataset of {} task...'.format(task))
    test_data = MRData(
        task,
        train=False,
        split='test',
        target_slices=target_slices,
        input_dim=image_size,
        data_root=data_root,
        label_root=label_root,
    )
    test_loader = data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.weights, val_data.weights, test_data.weights
