import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


class FewShotSVHN(torch.utils.data.Dataset):
    def __init__(self, root, train=True, k_shot=10, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform

        import scipy.io as sio
        if train:
            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(self.root, 'train_32x32.mat'))
        else:
            loaded_mat = sio.loadmat(os.path.join(self.root, 'test_32x32.mat'))

        self.data = loaded_mat["X"]
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()
        np.place(self.labels, self.labels == 10, 0)
        print("labels: ", self.labels[:30])
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        self.sample_idx = []
        for i in range(10):
            cnt = 0
            for idx in range(len(self.labels)):
                if int(self.labels[idx]) == i and cnt < k_shot:
                    self.sample_idx.append(idx)
                    cnt += 1
            
            print(f"num of smaples added: {len(self.sample_idx)}.")

        np.random.shuffle(self.sample_idx)
        print(f"Totol {len(self.sample_idx)} samples.")

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, index):
        data_idx = self.sample_idx[index]
        img, label = self.data[data_idx], int(self.labels[data_idx])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, label


if __name__ == '__main__':
    root_dir = '/Users/xushijian/Desktop/xushijian/OOD/data/SVHN'
    dataset = FewShotSVHN(root=root_dir, train=True, k_shot=10,
                    transform=transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))

    cnt = np.zeros(10)
    for item in dataset:
        cnt[item[1]] += 1

    print(cnt)