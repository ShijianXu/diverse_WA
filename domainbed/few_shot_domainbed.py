import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


PACS_DOMAINS = {0:"art_painting", 1:"cartoon", 2:"photo", 3:"sketch"}
VLCS_DOMAINS = {0:"Caltech101", 1:"LabelMe", 2:"SUN09", 3:"VOC2007"}
PACS_CLASS_NUM = 7
VLCS_CLASS_NUM = 5


def get_domain(dataset, domain):
    if dataset == 'PACS':
        return PACS_DOMAINS[domain]
    elif dataset == 'VLCS':
        return VLCS_DOMAINS[domain]


def get_class_num(dataset):
    if dataset == 'PACS':
        return PACS_CLASS_NUM
    elif dataset == 'VLCS':
        return VLCS_CLASS_NUM


class CustomDataset(Dataset):
    def __init__(self, instances, image_dir, train=True) -> None:
        super().__init__()

        self.samples = instances
        self.is_train = train
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


class FewShotDomainBed():
    def __init__(self, root, dataset, domain, k_shot=10) -> None:
        super().__init__()

        self.root = root
        self.domain = get_domain(dataset, domain)
        self.image_dir = os.path.join(self.root, self.domain)
        self.class_num = get_class_num(dataset)

        labels_file = os.path.join(self.image_dir, 'image_list.txt')

        with open(labels_file, "r") as fp:
            content = fp.readlines()
        mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))
        print(f"Total {len(mapping)} images for domain {domain}.")

        np.random.shuffle(mapping)
        self.few_samples = []
        for i in range(self.class_num):
            cnt = 0
            for idx in range(len(mapping)):
                _, label = mapping[idx]
                if label == i and cnt < k_shot:
                    self.few_samples.append(mapping[idx])
                    cnt += 1

            print(f"num of adapt samples added: {len(self.few_samples)}")

        self.test_samples = list(set(mapping) - set(self.few_samples))
        print(f"Total {len(self.few_samples)} adaptation samples.")
        print(f"Total {len(self.test_samples)} test samples.")

        self.train_dataset = CustomDataset(self.few_samples, self.image_dir, train=True)
        self.test_dataset = CustomDataset(self.test_samples, self.image_dir, train=False)


if __name__ == '__main__':
    root_dir = '/Users/xushijian/Desktop/xushijian/OOD/data/PACS'
    dataset = FewShotDomainBed(root=root_dir, dataset='PACS', domain=3, k_shot=5)