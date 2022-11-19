# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torchvision.utils import save_image
from torchvision import transforms

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')

    # two models
    parser.add_argument('--hparams_seed1', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--hparams_seed2', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')

    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=20001,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    ## DiWA ##
    parser.add_argument('--init_step', action='store_true')
    parser.add_argument('--path_for_init', type=str, default=None)
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

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

    # hparams_1 for model_1
    if args.hparams_seed1 == 0:
        hparams_1 = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams_1 = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed1, args.trial_seed))

    # hparams_2 for model_2
    if args.hparams_seed2 == 0:
        hparams_2 = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams_2 = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed2, args.trial_seed))
    
    if args.hparams:
        hparams_1.update(json.loads(args.hparams))
        hparams_2.update(json.loads(args.hparams))

    print('HParams_1:')
    for k, v in sorted(hparams_1.items()):
        print('\t{}: {}'.format(k, v))

    print('HParams_2:')
    for k, v in sorted(hparams_2.items()):
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

    # TODO
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augment_transform = transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    num_workers = 4

    train_dataset = torchvision.datasets.CIFAR100(root="../data", train=True, download=True, transform=augment_transform)
    test_dataset = torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=test_transform)

    train_loaders = [InfiniteDataLoader(
        dataset=train_dataset,
        weights=None,
        batch_size=hparams_1['batch_size'],
        num_workers=num_workers)]

    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=num_workers)

    print(f"num train loaders: {len(train_loaders)}")

    eval_loader_names = ["train"]
    print("eval_loader_names: ", eval_loader_names)


    INPUT_SHAPE = (3, 224, 224)
    N_CLASSES = 100
    N_DOMAINS = 1
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    if args.algorithm == "ERM_2":
        algorithm = algorithm_class(INPUT_SHAPE, N_CLASSES,
            N_DOMAINS, hparams_1, hparams_2,
            init_step=args.init_step,
            path_for_init=args.path_for_init,
            device=device)
    else:
        raise NotImplementedError

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = len(train_dataset)/hparams_1['batch_size']

    n_steps = args.steps
    print("==> n_steps: ", n_steps)
    checkpoint_freq = args.checkpoint_freq

    def save_checkpoint(filename, results=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": INPUT_SHAPE,
            "model_num_classes": N_CLASSES,
            "model_num_domains": N_DOMAINS,
            "model_hparams1": hparams_1,
            "model_hparams2": hparams_2,
            "model_dict": algorithm.state_dict()
        }
        ## DiWA ##
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    best_score_1 = -float("inf")
    best_score_2 = -float("inf")
    last_results_keys = None

    # training
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        
        # update
        # {'loss1': loss1.item(), 'loss2': loss2.item()}
        step_vals = algorithm.update(minibatches_device, None)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        if step % checkpoint_freq == 0:
            print(step_vals)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # every few iterations, do evaluation
            acc_1, acc_2 = misc.accuracy_2(algorithm, eval_loader, None, device)
            results['eval_acc_1'] = acc_1
            results['eval_acc_2'] = acc_2

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams_1': hparams_1,
                'hparams_2': hparams_2,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")


            ## DiWA ##
            # current_score_1, current_score_2 = misc.get_score_2(results, args.test_envs)
            # if current_score_1 > best_score_1:
            #     best_score_1 = current_score_1
                
            #     print(f"Saving new best score at step: {step} at path: model_best.pkl")
            #     save_checkpoint(
            #         'model_best.pkl',
            #         results=json.dumps(results, sort_keys=True),
            #     )
            #     algorithm.to(device)

            # algorithm_dict = algorithm.state_dict()
            # start_step = step + 1
            # checkpoint_vals = collections.defaultdict(lambda: [])

            # if args.save_model_every_checkpoint:
            #     save_checkpoint(f'model_step{step}.pkl')

    ## DiWA ##
    if args.init_step:
        algorithm.save_path_for_future_init(args.path_for_init)
    
    save_checkpoint('model.pkl')
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
