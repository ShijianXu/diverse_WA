import os
import torch
from torchvision import datasets, transforms

import numpy as np

from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.few_shot_model import Adaptor, DiWA_Adaptor

# a single resnet18 well-trained on mnist, without adaptation
# model = 'mnist_res18_imagenet_sweep_diwa_sam/028636adf680f19f6dde3544536412a0/model_best.pkl'
# model = 'usps_res18_imagenet_sweep_diwa_sam/ca208d518eaa13fa7726eef1bcdeea06/model_best.pkl'

# resnet18, imagenet pretrained, weight averaged then 10-shot adapt on mnist-m
model = 'usps_res18_imagenet_sam_adapt_2_mnist_10_shot/adapted_model.pkl'
hparams = {}
hparams['lr'] = 5e-4
hparams['rho'] = 0.05
hparams['weight_decay'] = 5e-4
hparams['batch_size'] = 8


save_dict = torch.load(model, map_location=torch.device('cpu'))

# hparams = save_dict["model_hparams"]

network = Adaptor(
        channels=3,
        num_classes=10,
        hparams=hparams,
        model_name='resnet18'
    )
state_dict = save_dict["model_dict"]

new_state_dict = {key.replace("network_wa.", "classifier."): value for key, value in state_dict.items()}  
missing_keys, unexpected_keys =  network.load_state_dict(new_state_dict, strict=False)

# missing_keys, unexpected_keys =  network.load_state_dict(state_dict, strict=False)
print(f"Load individual model with missing keys {missing_keys} and unexpected keys {unexpected_keys}.")

# test_dataset  = few_shot_datasets.get_dataset('../data', 'MNISTM', 64, False)
# test_dataset  = few_shot_datasets.get_dataset('../data', 'SVHN', 64, False)
test_dataset  = few_shot_datasets.get_dataset('../data', 'MNIST', 64, False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

features = []
labels = []
network.eval()
correct = 0
total = 0

cnt = 0
for x, y in test_dataloader:
    print(cnt)
    if cnt > 1000:
        break

    p = network.predict(x)
    correct += (p.argmax(1).eq(y).float()).sum().item()
    total += len(x)

    features.append(p.detach().numpy())
    labels.append(y)

    cnt += len(y)

print("accuracy: ", correct / total)

# Concatenate the features and labels
features = np.concatenate(features)
labels = np.concatenate(labels)
print(features.shape)
print(labels.shape)

with open('feature_usps_2_mnist_adapt_after_wa.npy', 'wb') as f:
    np.save(f, features)
with open('label_usps_2_mnist_adapt_after_wa.npy', 'wb') as f:
    np.save(f, labels)