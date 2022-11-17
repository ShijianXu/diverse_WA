## code used to load and test a single model

"""
python3 -m domainbed.scripts.test_single \
       --data_dir=../data \
       --model_path=./model_best.pkl \
       --dataset CIFAR100 \
       --test_env 0
"""


import argparse
import os
import json
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import transforms

from domainbed import datasets, algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_env', type=int)

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str, default="default")

    inf_args = parser.parse_args()
    return inf_args


def create_splits(domain, inf_args, dataset, _filter):
    splits = []

    for env_i, env in enumerate(dataset):
        if domain == "test" and env_i != inf_args.test_env:
            continue
        elif domain == "train" and env_i == inf_args.test_env:
            continue

        if _filter == "full":
            splits.append(env)
        else:
            raise NotImplementedError

    return splits


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Begin single model test for: {inf_args} with device: {device}")

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(inf_args.data_dir, [inf_args.test_env], hparams={"data_augmentation": False})
    elif inf_args.dataset == 'CIFAR100':
        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=test_transform)
    else:
        raise NotImplementedError


    # load test domain data
    if inf_args.dataset in vars(datasets):
        data_splits, data_names = [], []
        dict_domian_to_filter = {"test": "full"}

        for domain in dict_domian_to_filter:
            _data_splits = create_splits(domain, inf_args, dataset, dict_domian_to_filter[domain])
            if domain == "train":
                raise NotImplementedError
            else:
                data_splits.append(_data_splits[0])

            data_names.append(domain)
    

    # start testing
    save_dict = torch.load(inf_args.model_path, map_location=torch.device(device))
    train_args = save_dict["args"]

    if inf_args.dataset in vars(datasets):
        algorithm = algorithms_inference.ERM(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
    else:
        INPUT_SHAPE = (3, 224, 224)
        N_CLASSES = 100
        N_DOMAINS = 1
        algorithm = algorithms_inference.ERM(
            INPUT_SHAPE,
            N_CLASSES,
            N_DOMAINS,
            save_dict["model_hparams"]
        )
    algorithm.load_state_dict(save_dict["model_dict"], strict=False)
    algorithm.to(device)
    algorithm.eval()

    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if inf_args.dataset in vars(datasets):
        data_loaders = [
            FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
            for split in data_splits
        ]
    
        data_evals = zip(data_names, data_loaders)

        for name, loader in data_evals:
            print(f"Inference at {name}")
            acc = misc.accuracy(algorithm, loader, None, device)

    else:
        eval_loader = FastDataLoader(
            dataset=dataset,
            batch_size=64,
            num_workers=4)
        acc = misc.accuracy(algorithm, eval_loader, None, device)

    print(f"Test accuracy: {acc}")


if __name__ == "__main__":
    main()