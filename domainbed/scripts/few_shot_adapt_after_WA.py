"""
python3 -m domainbed.scripts.few_shot_adapt_after_WA \
    --data_dir=../data \
    --model_name resnet50 \
    --target_dataset VisDA \
    --target_domain 0 \
        # `target_domain` only used for PACS and VLCS `target_dataset`
    --num_classes 12 \
    --sweep_dir=./VisDA_sweep_diwa_adam \
    --output_dir=./VisDA_adam_diwa_synth2real_adam_10_shot \
    --weight_selection uniform \
    --opt_name Adam \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 100 \
    --test_freq 10
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
from domainbed.few_shot_model import Adaptor, DiWA_Adaptor, ERM_2


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

    hparams = {}
    hparams['lr'] = 5e-4
    hparams['rho'] = args.sam_rho
    hparams['weight_decay'] = 5e-4
    hparams['batch_size'] = 8
    
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')


def _get_args():
    parser = argparse.ArgumentParser(description="Diverse Weight Averaging for Adaptation")
    parser.add_argument('--sweep_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--opt_name', type=str, default="Adam")
    parser.add_argument('--weight_selection', type=str, default="uniform")
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--k_shot', type=int, default=10)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--test_freq', type=int, default=-1)

    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--target_dataset', type=str)
    parser.add_argument('--model_name', type=str, default="resnet18")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--target_domain', type=int, default=None)      # only set for PACS & VLCS target_dataset
    
    # indicate whether the sweeped model is grad_diverse trained
    # if set, then each checkpoint contains two model, should handled separately
    # currently only PACS and VLCS
    parser.add_argument('--diverse', action='store_true')

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
        if os.path.isdir(output_folder) and "done" in os.listdir(output_folder) and "model.pkl" in os.listdir(output_folder)
    ]

    model_folders_list = []
    for folder in output_folders:
        model_folders_list.append(folder)

    return model_folders_list


def get_wa_model(ckpt_folders, hparams, args, device):
    wa_adaptor = DiWA_Adaptor(
        channels=3,
        num_classes=args.num_classes,
        hparams=hparams,
        opt_name=args.opt_name
    )

    for folder in ckpt_folders:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"), map_location=torch.device(device))
        train_args = save_dict["args"]
        hparams = save_dict["model_hparams"]

        # load individual weights
        ind_adaptor = Adaptor(
            channels=3,
            num_classes=args.num_classes,
            hparams=hparams,
            model_name=args.model_name
        )
        missing_keys, unexpected_keys =  ind_adaptor.load_state_dict(save_dict["model_dict"], strict=False)
        print(f"Load individual model with missing keys {missing_keys} and unexpected keys {unexpected_keys}.")

        wa_adaptor.add_weights(ind_adaptor.classifier)
        del ind_adaptor
    
    wa_adaptor.set_optimizer()

    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return wa_adaptor


def get_wa_model_diverse(ckpt_folders, hparams, args, device):
    wa_adaptor = DiWA_Adaptor(
        channels=3,
        num_classes=args.num_classes,
        hparams=hparams,
        opt_name=args.opt_name
    )

    for folder in ckpt_folders:
        save_dict = torch.load(os.path.join(folder, "model.pkl"), map_location=torch.device(device))
        train_args = save_dict["args"]

        # load individual weights
        algorithm = ERM_2(
            args.num_classes,
            save_dict["model_hparams1"],
            save_dict["model_hparams2"],
        )
        missing_keys, unexpected_keys = algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        print(f"Loading individual models with missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}.")
        
        wa_adaptor.add_weights(algorithm.network1)
        wa_adaptor.add_weights(algorithm.network2)
        del algorithm
    
    wa_adaptor.set_optimizer()

    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return wa_adaptor


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Begin DiWA for: {inf_args} with device: {device}")

    model_folders = get_model_folders(inf_args)
    print(model_folders)

    hparams = {}
    # hparams['lr'] = 5e-4
    hparams['lr'] = 5e-5        # smaller lr ???
    
    hparams['rho'] = inf_args.sam_rho
    hparams['weight_decay'] = 5e-4
    hparams['batch_size'] = 8

    if inf_args.weight_selection == "uniform":
        if inf_args.diverse:
            wa_model = get_wa_model_diverse(model_folders, hparams, inf_args, device)
        else:
            wa_model = get_wa_model(model_folders, hparams, inf_args, device)

        # print("Get Averaged Model. Test on the source domain.")
        # source_test(wa_model, inf_args)

        print("Get Averaged Model. Ready to adapt.")
        adapt(wa_model, inf_args)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()