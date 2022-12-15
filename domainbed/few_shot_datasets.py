import os
import os.path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision

from domainbed import mnist_m

#========================================
#                 utils
#========================================
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_valid_file(x: str) -> bool:
    return has_file_allowed_extension(x, IMG_EXTENSIONS)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
    
#===============================================


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
]


class CustomDataset(Dataset):
    def __init__(self, instances, train=True) -> None:
        super().__init__()

        self.samples = instances
        self.loader = default_loader
        self.is_train = train

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

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.is_train:
            sample = self.augment_transform(sample)
        else:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


class SingleEnvironmentDatasets():
    def __init__(self, root, test_env, k_shot=10):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        for i in range(len(environments)):
            print(f"env_{i}:", environments[i])

        print(test_env)
        assert len(test_env)==1
        environment = environments[test_env[0]]


        direct = os.path.join(root, environment)
        classes, class_to_idx = self.find_classes(direct)

        self.train_samples = []
        self.test_samples = []

        for target_class in sorted(class_to_idx.keys()):
            instances = []
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(direct, target_class)
            
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                np.random.shuffle(instances)
                self.train_samples.extend(instances[:k_shot])
                self.test_samples.extend(instances[k_shot:])

        self.train_dataset = CustomDataset(self.train_samples, train=True)
        self.test_dataset = CustomDataset(self.test_samples, train=False)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(classes)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)


class VLCS(SingleEnvironmentDatasets):
    CHECKPOINT_FREQ = 50 ## DiWA ##
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, k_shot=10):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, k_shot)


class PACS(SingleEnvironmentDatasets):
    CHECKPOINT_FREQ = 100 ## DiWA ##
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, k_shot=10):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, k_shot)


## Datasets for MNIST and MNIST-M

def get_dataset(root, data_name, imsize, train=True):
    if data_name == 'MNIST':
        if train:
            print("Return MNIST train.")
            return torchvision.datasets.MNIST(
                root=root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(imsize),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        else:
            print("Return MNIST test.")
            return torchvision.datasets.MNIST(
                root=root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(imsize),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    elif data_name == 'MNISTM':
        if train:
            print("Return MNIST-M train.")
            return mnist_m.MNIST_M(
                root=root+'/mnist_m', train=True,
                transform=transforms.Compose([
                    transforms.Resize(imsize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        else:
            print("Return MNIST-M test.")
            return mnist_m.MNIST_M(
                root=root+'/mnist_m', train=False,
                transform=transforms.Compose([
                    transforms.Resize(imsize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))

    else:
        raise NotImplementedError