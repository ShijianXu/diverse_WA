"""
python3 -m domainbed.scripts.few_shot_adapt_baseline \
    --data_dir=../data \
    --model_name resnet50 \
    --target_dataset VisDA \
    --target_domain 0 \
        # `target_domain` only used for PACS and VLCS `target_dataset`
    --num_classes 12 \
    --model_path=./PACS_0_baseline_sam_rho_0_05/model_best.pkl \
    --output_dir=./PACS_0_baseline_sam_rho_0_05 \
    --opt_name Adam \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 2000 \
    --test_freq 10


python3 -m domainbed.scripts.few_shot_adapt_baseline \
    --data_dir=../data \
    --model_name resnet50 \
    --target_dataset PACS \
    --target_domain 0 \
    --num_classes 7 \
    --model_path=./PACS_0_baseline_sam_rho_0_05/model_best.pkl \
    --output_dir=./PACS_0_baseline_sam_rho_0_05 \
    --opt_name SAM \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 2000 \
    --test_freq 10

"""

import argparse
import os
import random
import sys

import numpy as np
import PIL
import torch
import torch.utils.data

from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.few_shot_model import Adaptor, DiWA_Adaptor


def source_test(adaptor, args):
    test_dataset  = few_shot_datasets.get_dataset(args.data_dir, 'MNIST', 64, False)
    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaptor.to(device)

    test_acc = misc.accuracy(adaptor, eval_loader, None, device)

    print("WA model test accuracy on MNIST: ", test_acc)
    

def adapt(adaptor, args):
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

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
    elif args.target_dataset == 'PACS' or args.target_dataset == 'VLCS':
        domainbed_dataset = few_shot_datasets.get_dataset(
            data_dir, args.target_dataset, 64, 
            k_shot=args.k_shot, 
            target_domain=args.target_domain)
        train_dataset = domainbed_dataset.train_dataset
        test_dataset = domainbed_dataset.test_dataset
    else:
        raise NotImplementedError

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=None,
        batch_size=8,
        num_workers=1)

    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=1)

    train_minibatches_iterator = zip(train_loader)
    n_steps = args.steps

    def save_checkpoint(filename, results=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_dict": adaptor.state_dict()
        }
        ## DiWA ##
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    # test before adaptation
    print("Testing ...")
    adaptor.eval()
    acc = misc.accuracy(adaptor, eval_loader, None, device)
    print("Before few shot adaptation, the accuracy is: ", acc)

    # training
    print("Training ...")
    print("Training steps: ", n_steps)
    adaptor.train()
    
    best_acc = 0
    start_step = 0
    for step in range(start_step, n_steps):
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
            
        # update
        step_vals = adaptor.update(minibatches_device)

        if args.test_freq != -1:
            if step % args.test_freq == 0:
                # test 
                print("Testing again ...")
                adaptor.eval()
                acc = misc.accuracy(adaptor, eval_loader, None, device)

                if acc > best_acc:
                    print(f"A better accuracy after {step} adaptation steps: {acc}.")
                    best_acc = acc
                    save_checkpoint('adapted_model.pkl')
    
    if args.test_freq == -1:
        print("Test again after adaptation ...")
        adaptor.eval()
        acc = misc.accuracy(adaptor, eval_loader, None, device)
        print(f"Accuracy after {n_steps} adaptation steps: {acc}.")
        save_checkpoint('adapted_model.pkl')

    with open(os.path.join(args.output_dir, 'adapt_done'), 'w') as f:
        f.write('done')


def _get_args():
    parser = argparse.ArgumentParser(description="Diverse Weight Averaging for Adaptation")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--opt_name', type=str, default="Adam")
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--k_shot', type=int, default=10)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--test_freq', type=int, default=-1)

    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--target_dataset', type=str)
    parser.add_argument('--model_name', type=str, default="resnet18")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--target_domain', type=int, default=None)      # only set for PACS & VLCS target_dataset

    inf_args = parser.parse_args()
    return inf_args


def get_model(hparams, args, device):
    
    save_dict = torch.load(args.model_path, map_location=torch.device(device))
    train_args = save_dict["args"]

    # load weights
    adaptor = Adaptor(
        channels=3,
        num_classes=args.num_classes,
        hparams=hparams,
        opt_name=args.opt_name,
        model_name=args.model_name
    )

    state_dict = save_dict["model_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('network.0.network', 'classifier')] = state_dict.pop(key)
    for key in list(state_dict.keys()):
        state_dict[key.replace('network.1', 'classifier.fc')] = state_dict.pop(key)

    missing_keys, unexpected_keys = adaptor.load_state_dict(state_dict, strict=False)
    print(f"Load model with missing keys {missing_keys} and unexpected keys {unexpected_keys}.")

    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return adaptor


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Begin Adaptation for: {inf_args} with device: {device}")

    hparams = {}
    # hparams['lr'] = 5e-4
    hparams['lr'] = 5e-5        # smaller lr ???
    
    hparams['rho'] = inf_args.sam_rho
    hparams['weight_decay'] = 5e-4
    hparams['batch_size'] = 8

    model = get_model(hparams, inf_args, device)
    print("Model loaded. Ready to adapt.")
    adapt(model, inf_args)


if __name__ == "__main__":
    main()