import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


class FewShotVisDA(torch.utils.data.Dataset):
    def __init__(self, root, train=True, k_shot=10, transform=None):
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
        mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

        self.samples = []
        for i in range(12):
            cnt = 0
            for idx in range(len(mapping)):
                _, label = mapping[idx]
                if label == i and cnt < k_shot:
                    self.samples.append(mapping[idx])
                    cnt += 1
            
            print(f"num of samples added: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = os.path.join(self.image_dir, image)
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label


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
    # dataset = FewShotVisDA(root=root_dir, train=True, k_shot=5,
    #                 transform=transforms.Compose([
    #                 transforms.Resize(256),
    #                 transforms.CenterCrop(224),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(
    #                     mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
    #             ]))

    dataset = VisDA(root=root_dir, train=False,
                    transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                ]))

    print(len(dataset))
    for i in range(30):
        item = dataset[i]
        print(item)