import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


class FewShotUSPS(torch.utils.data.Dataset):
    def __init__(self, root, train=True, k_shot=10, transform=None) -> None:
        super().__init__()

        self.root = root
        self.transform = transform

        import bz2
        if train:
            full_path = os.path.join(self.root, 'usps.bz2')
        else:
            full_path = os.path.join(self.root, 'usps.t.bz2')

        with bz2.open(full_path) as fp:
            raw_data = [line.decode().split() for line in fp.readlines()]
            tmp_list = [[x.split(":")[-1] for x in data[1:]] for data in raw_data]
            imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
            imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
            targets = [int(d[0]) - 1 for d in raw_data]

        self.data = imgs
        self.targets = targets

        self.sample_idx = []
        for i in range(10):
            cnt = 0
            for idx in range(len(self.targets)):
                if int(self.targets[idx]) == i and cnt < k_shot:
                    self.sample_idx.append(idx)
                    cnt += 1
            
            print(f"num of smaples added: {len(self.sample_idx)}.")

        np.random.shuffle(self.sample_idx)
        print(f"Totol {len(self.sample_idx)} samples.")


    def __getitem__(self, index):
        data_idx = self.sample_idx[index]
        img, target = self.data[data_idx], int(self.targets[data_idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.sample_idx)


if __name__ == '__main__':
    root_dir = '/Users/xushijian/Desktop/xushijian/OOD/data/USPS'
    dataset = FewShotUSPS(root=root_dir, train=True, k_shot=5,
                    transform=transforms.Compose([
                    transforms.Resize(64),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))

    cnt = np.zeros(10)
    for item in dataset:
        cnt[item[1]] += 1

    print(cnt)