# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
# adaptation and test
python3 -m domainbed.scripts.few_shot_adapt \
       --data_dir=../data \
       --output_dir=./PACS_0_10_shot_erm_adapt \
       --algorithm ERM \
       --dataset PACS \
       --test_env 0 \
       --k_shot 10 \
       --model_path /scratch/izar/sxu/PACS_0_baseline_sam_rho_0_01/model_best.pkl \
       --steps 100

python3 -m domainbed.scripts.few_shot_adapt \
       --data_dir=../data \
       --output_dir=./PACS_0_10_shot_sam_rho_0_01_adapt \
       --algorithm SAM \
       --sam_rho 0.01 \
       --dataset PACS \
       --test_env 0 \
       --k_shot 10 \
       --model_path /scratch/izar/sxu/PACS_test_0_init_sam_rho_0.01.pth \
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

from domainbed import few_shot_datasets
from domainbed import hparams_registry
from domainbed import algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    ## DiWA ##
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--k_shot', type=int, default=10)
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(few_shot_datasets):
        dataset = vars(few_shot_datasets)[args.dataset](args.data_dir, args.test_envs, args.k_shot)
        # e.g. PACS(root, test_envs, k_shot)
    else:
        raise NotImplementedError

    train_loader = InfiniteDataLoader(
        dataset=dataset.train_dataset,
        weights=None,
        batch_size=8,
        num_workers=2)

    eval_loader = FastDataLoader(
        dataset=dataset.test_dataset,
        batch_size=64,
        num_workers=4
    )

    # load checkpoint before testing
    save_dict = torch.load(args.model_path, map_location=torch.device(device))
    
    if args.algorithm == "ERM":
        algorithm = algorithms_inference.ERM(
            dataset.input_shape,
            dataset.num_classes,
            num_domains=1, 
            hparams=save_dict["model_hparams"]
        )
    else:
        raise NotImplementedError

    algorithm.load_state_dict(save_dict["model_dict"], strict=False)
    algorithm.to(device)

    train_minibatches_iterator = zip(train_loader)
    # checkpoint_vals = collections.defaultdict(lambda: [])

    n_steps = args.steps
    print("==> n_steps: ", n_steps)
    # checkpoint_freq = args.checkpoint_freq

    def save_checkpoint(filename, results=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        ## DiWA ##
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    best_score = -float("inf")
    last_results_keys = None


    # test before adaptation
    print("Testing ...")
    algorithm.eval()
    acc = misc.accuracy(algorithm, eval_loader, None, device)
    print("Before few shot adaptation, the accuracy is: ", acc)

    # training
    print("Training ...")
    algorithm.train()
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
            
        # update
        step_vals = algorithm.update(minibatches_device, None)

    # test 
    print("Testing again ...")
    algorithm.eval()
    acc = misc.accuracy(algorithm, eval_loader, None, device)
    print("After few shot adaptation, the accuracy is: ", acc)

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
