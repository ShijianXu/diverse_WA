"""
# train
python3 -m domainbed.scripts.few_shot_adapt_mnistm \
    --data_dir=../data \
    --opt_name Adam \
    --output_dir=./mnist_adam_2_mnistm_adam_5_shot \
    --model_path /scratch/izar/sxu/mnist_pretrain_Adam/model_best.pkl \
    --k_shot 10 \
    --steps 100
"""

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import mnist_m
from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.few_shot_mnist_net import Adaptor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Domain Adaptation MNIST-M Adapt")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--opt_name', type=str, default="Adam")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default="MNIST_2_MNISTM")
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--k_shot', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    hparams = {}
    hparams['lr'] = 5e-4
    hparams['rho'] = args.sam_rho
    hparams['weight_decay'] = 5e-4
    hparams['batch_size'] = 8
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data_dir = os.path.abspath(args.data_dir)

    train_dataset = few_shot_datasets.get_dataset(data_dir, 'MNISTM', 64, True, args.k_shot)
    test_dataset  = few_shot_datasets.get_dataset(data_dir, 'MNISTM', 64, False)

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=None,
        batch_size=8,
        num_workers=2)

    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=2)

    save_dict = torch.load(args.model_path, map_location=torch.device(device))
    adaptor = Adaptor(channels=3, num_classes=10, hparams=hparams, opt_name=args.opt_name)
    
    adaptor.load_state_dict(save_dict["model_dict"], strict=False)
    adaptor.to(device)

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
    adaptor.train()
    
    start_step = 0
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
            
        # update
        step_vals = adaptor.update(minibatches_device, None)

    # test 
    print("Testing again ...")
    adaptor.eval()
    acc = misc.accuracy(adaptor, eval_loader, None, device)
    print("After few shot adaptation, the accuracy is: ", acc)

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')