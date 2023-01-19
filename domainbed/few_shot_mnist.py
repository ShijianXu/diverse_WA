import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import codecs
import matplotlib.pyplot as plt

class FewShotMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, k_shot=10, transform=None):
        super().__init__()

        self.root = root
        self.raw_folder = os.path.join(self.root, 'MNIST', 'raw')
        self.transform = transform
        self.train = train

        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        self.data = read_image_file(os.path.join(self.raw_folder, image_file))
        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        self.targets = read_label_file(os.path.join(self.raw_folder, label_file))
        
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


    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, index):
        data_idx = self.sample_idx[index]
        img, label = self.data[data_idx], int(self.targets[data_idx])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = torch.iinfo(torch_type).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with torch.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 3:
        raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
    return x