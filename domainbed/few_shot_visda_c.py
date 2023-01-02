import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, instances, image_dir, train=True) -> None:
        super().__init__()

        self.samples = instances
        self.is_train = train
        self.image_dir = image_dir

        self.transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        self.augment_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = os.path.join(self.image_dir, image)

        if self.is_train:
            image = self.augment_transform(Image.open(image).convert('RGB'))
        else:
            image = self.transform(Image.open(image).convert('RGB'))

        return image, label

    def __len__(self):
        return len(self.samples)


class FewShotVisDA():
    def __init__(self, root, k_shot=10):
        super().__init__()

        self.root = root

        self.image_dir = os.path.join(root, 'validation')
        labels_file = os.path.join(root, 'validation', 'image_list.txt')

        with open(labels_file, "r") as fp:
            content = fp.readlines()
        mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))
        print(f"Total {len(mapping)} validation samples.")

        self.few_samples = []
        for i in range(12):
            cnt = 0
            for idx in range(len(mapping)):
                _, label = mapping[idx]
                if label == i and cnt < k_shot:
                    self.few_samples.append(mapping[idx])
                    cnt += 1
            
            print(f"num of adaptation samples added: {len(self.few_samples)}")
        
        self.test_samples = list(set(mapping) - set(self.few_samples))
        print(f"Total {len(self.few_samples)} adaptation samples.")
        print(f"Total {len(self.test_samples)} test samples.")

        self.train_dataset = CustomDataset(self.few_samples, self.image_dir, train=True)
        self.test_dataset = CustomDataset(self.test_samples, self.image_dir, train=False)


class VisDA(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform

        if train:
            self.image_dir = os.path.join(root, 'train')
            labels_file = os.path.join(root, 'train', 'image_list.txt')
        else:
            self.image_dir = os.path.join(root, 'validation')
            labels_file = os.path.join(root, 'validation', 'image_list.txt')

        with open(labels_file, "r") as fp:
            content = fp.readlines()
        self.mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image, label = self.mapping[idx]
        image = os.path.join(self.image_dir, image)
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label


if __name__ == '__main__':
    root_dir = '/Users/xushijian/Desktop/xushijian/OOD/data/VisDA'
    dataset = FewShotVisDA(root=root_dir, k_shot=10)

    # dataset = VisDA(root=root_dir, train=False,
    #                 transform=transforms.Compose([
    #                 transforms.Resize(256),
    #                 transforms.CenterCrop(224),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(
    #                     mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
    #             ]))

    # print(len(dataset))
    # for i in range(30):
    #     item = dataset[i]
    #     print(item)