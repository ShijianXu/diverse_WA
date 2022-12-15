import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


class MNIST_M(torch.utils.data.Dataset):
	def __init__(self, root, train, transform=None):
		self.train = train
		self.transform = transform
		if train:
			self.image_dir = os.path.join(root, 'mnist_m_train')
			labels_file = os.path.join(root, "mnist_m_train_labels.txt")
		else:
			self.image_dir = os.path.join(root, 'mnist_m_test')
			labels_file = os.path.join(root, "mnist_m_test_labels.txt")

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


class FewShotMNIST_M(torch.utils.data.Dataset):
	def __init__(self, root, train=True, k_shot=10, transform=None):
		super().__init__()

		self.transform = transform

		if train:
			self.image_dir = os.path.join(root, 'mnist_m_train')
		else:
			self.image_dir = os.path.join(root, 'mnist_m_test')

		self.samples = []
		for i in range(10):
			label_i_file = os.path.join(root, f"mnist_m_train_label_{i}.txt")
			with open(label_i_file, "r") as fp:
				content = fp.readlines()

			mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

			np.random.shuffle(mapping)
			self.samples.extend(mapping[:k_shot])

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		image, label = self.samples[index]
		image = os.path.join(self.image_dir, image)
		
		assert self.transform is not None
		image = self.transform(Image.open(image).convert('RGB'))
		return image, label


if __name__ == '__main__':
	root_dir = '/Users/xushijian/Desktop/xushijian/OOD/data/mnist_m'
	dataset = FewShotMNIST_M(root_dir, True, k_shot=10,
                transform=transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))