"""
python3 -m domainbed.scripts.few_shot_adapt_before_WA \
    --data_dir=../data \
    --model_name resnet50 \
    --target_dataset VisDA \
    --num_classes 12 \
    --sweep_dir=./VisDA_sweep_diwa_adam \
    --weight_selection uniform \
    --opt_name Adam \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 100
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.few_shot_model import Adaptor, DiWA_Adaptor

def _get_args():
    parser = argparse.ArgumentParser(description="Diverse Weight Averaging after Individual Adaptation")
    parser.add_argument('--sweep_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--opt_name', type=str, default="Adam")
    parser.add_argument('--weight_selection', type=str, default="uniform")
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--k_shot', type=int, default=10)
    parser.add_argument('--steps', type=int, default=100)

    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--target_dataset', type=str)
    parser.add_argument('--model_name', type=str, default="resnet18")
    parser.add_argument('--num_classes', type=int, default=10)

    inf_args = parser.parse_args()
    return inf_args


def get_model_folders(inf_args):
    output_folders = [
        os.path.join(sweep_dir, path)
        for sweep_dir in inf_args.sweep_dir.split(",")
        for path in os.listdir(sweep_dir)
    ]

    output_folders = [
        output_folder for output_folder in output_folders
        if os.path.isdir(output_folder) and "done" in os.listdir(output_folder) and "model_best.pkl" in os.listdir(output_folder)
    ]

    model_folders_list = []
    for folder in output_folders:
        model_folders_list.append(folder)

    return model_folders_list


def adapt(adaptor, ckpt_folder, hparams, train_args, args):
    print("\nIndividual model from: ", ckpt_folder)

    random.seed(train_args['seed'])
    np.random.seed(train_args['seed'])
    torch.manual_seed(train_args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaptor.to(device)

    data_dir = os.path.abspath(args.data_dir)

    if args.target_dataset == 'MNIST':
        train_dataset = few_shot_datasets.get_dataset(data_dir, 'MNIST', 64, True, args.k_shot)
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'MNIST', 64, False)
    elif args.target_dataset == 'MNISTM':
        train_dataset = few_shot_datasets.get_dataset(data_dir, 'MNISTM', 64, True, args.k_shot)
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'MNISTM', 64, False)
    elif args.target_dataset == 'SVHN':
        train_dataset = few_shot_datasets.get_dataset(data_dir, 'SVHN', 64, True, args.k_shot)
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'SVHN', 64, False)
    elif args.target_dataset == 'USPS':
        train_dataset = few_shot_datasets.get_dataset(data_dir, 'USPS', 64, True, args.k_shot)
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'USPS', 64, False)
    elif args.target_dataset == 'VisDA':
        visda_dataset = few_shot_datasets.get_dataset(data_dir, 'VisDA_few_shot', k_shot=args.k_shot)
        train_dataset = visda_dataset.train_dataset
        test_dataset = visda_dataset.test_dataset
    else:
        raise NotImplementedError

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=None,
        batch_size=8,
        num_workers=2)

    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=2)

    train_minibatches_iterator = zip(train_loader)
    n_steps = args.steps

    def save_checkpoint(filename, results=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_hparams": hparams,
            "model_dict": adaptor.state_dict()
        }
        ## DiWA ##
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(ckpt_folder, filename))

    # test before adaptation
    print("Testing ...")
    adaptor.eval()
    acc_before = misc.accuracy(adaptor, eval_loader, None, device)
    print("Before few shot adaptation, the accuracy is: ", acc_before)

    # training
    print("Training ...")
    print("Training steps: ", n_steps)
    adaptor.train()
    
    start_step = 0
    for step in range(start_step, n_steps):
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
            
        # update
        step_vals = adaptor.update(minibatches_device)

    # test 
    print("Testing again ...")
    adaptor.eval()
    acc_after = misc.accuracy(adaptor, eval_loader, None, device)
    print("After few shot adaptation, the accuracy is: ", acc_after)

    save_checkpoint('adapted_model.pkl')

    return acc_before, acc_after


def individual_adapt(ckpt_folders, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accs_before = {}
    accs_after = {}
    for folder in ckpt_folders:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"), map_location=torch.device(device))
        train_args = save_dict["args"]

        hparams = save_dict["model_hparams"]
        print("Hparams: ", hparams)

        # load individual weights
        ind_adaptor = Adaptor(
            channels=3,
            num_classes=args.num_classes,
            hparams=hparams,
            opt_name=train_args['opt_name'],
            model_name=args.model_name
        )
        missing_keys, unexpected_keys =  ind_adaptor.load_state_dict(save_dict["model_dict"], strict=False)
        print(f"Load individual model with missing keys {missing_keys} and unexpected keys {unexpected_keys}.")

        acc_before, acc_after = adapt(ind_adaptor, folder, hparams, train_args, args)
        accs_before[folder] = acc_before
        accs_after[folder] = acc_after
    
    print("Accuracies before adaptation")
    for k, v in accs_before.items():
        print('\t{}" {}'.format(k, v))
    print(f"The mean accuracy: ", np.mean(list(accs_before.values())))

    print("Accuracies after adaptation")
    for k, v in accs_after.items():
        print('\t{}" {}'.format(k, v))
    print(f"The mean accuracy: ", np.mean(list(accs_after.values())))


def wa_test(ckpt_folders, inf_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wa_adaptor = DiWA_Adaptor(
        channels=3,
        num_classes=inf_args.num_classes,
        hparams=None
    )

    for folder in ckpt_folders:
        save_dict = torch.load(os.path.join(folder, "adapted_model.pkl"), map_location=torch.device(device))
        train_args = save_dict["args"]
        hparams = save_dict["model_hparams"]

        # load individual weights
        ind_adaptor = Adaptor(
            channels=3,
            num_classes=inf_args.num_classes,
            hparams=hparams,
            model_name=inf_args.model_name
        )
        missing_keys, unexpected_keys =  ind_adaptor.load_state_dict(save_dict["model_dict"], strict=False)
        print(f"Load adapted model with missing keys {missing_keys} and unexpected keys {unexpected_keys}.")

        wa_adaptor.add_weights(ind_adaptor.classifier)
        del ind_adaptor

    data_dir = os.path.abspath(inf_args.data_dir)

    if inf_args.target_dataset == 'MNIST':
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'MNIST', 64, False)
    elif inf_args.target_dataset == 'MNISTM':
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'MNISTM', 64, False)
    elif inf_args.target_dataset == 'SVHN':
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'SVHN', 64, False)
    elif inf_args.target_dataset == 'USPS':
        test_dataset  = few_shot_datasets.get_dataset(data_dir, 'USPS', 64, False)
    elif inf_args.target_dataset == 'VisDA':
        visda_dataset = few_shot_datasets.get_dataset(data_dir, 'VisDA_few_shot')
        test_dataset = visda_dataset.test_dataset
    else:
        raise NotImplementedError

    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=2)

    wa_adaptor.to(device)
    print("Test the averaged adapted model.")
    acc = misc.accuracy(wa_adaptor, eval_loader, None, device)
    print("Adaptation before weight averaging, the accuracy is: ", acc)


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Begin DiWA for: {inf_args} with device: {device}")

    model_folders = get_model_folders(inf_args)
    print(f"Total {len(model_folders)} models.")
    print(model_folders)

    individual_adapt(model_folders, inf_args)

    wa_test(model_folders, inf_args)


if __name__ == "__main__":
    main()