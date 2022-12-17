"""
# train baseline
python3 -m domainbed.scripts.few_shot_mnist_train \
    --data_dir=../data \
    --opt_name Adam \
    --output_dir=./mnist_pretrain \
    --path_for_init ./mnist_future_init.pth \
    --steps 500 \
    --check_freq 5 \
    --init_step

"""


import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from domainbed import mnist_m
from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed import hparams_mnist_registry
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.few_shot_mnist_net import Adaptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptation MNIST Pretrain")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--opt_name', type=str, default="Adam")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--check_freq', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="MNIST_pretrain")
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--path_for_init', type=str, default=None)
    parser.add_argument('--init_step', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))


    # set hyperparameters
    if args.hparams_seed == 0:
        hparams = hparams_mnist_registry.default_hparams(args.opt_name)
    else:
        hparams = hparams_mnist_registry.random_hparams(args.opt_name, misc.seed_hash(args.hparams_seed, args.trial_seed))

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

    data_dir = os.path.abspath(args.data_dir)

    train_dataset = few_shot_datasets.get_dataset(data_dir, 'MNIST', 64, True)
    test_dataset  = few_shot_datasets.get_dataset(data_dir, 'MNIST', 64, False)

    # train_dataset = None
    # test_dataset  = few_shot_datasets.get_dataset(data_dir, 'MNISTM', 64, False)

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=None,
        batch_size=hparams['batch_size'],
        num_workers=2)

    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=hparams['batch_size'],
        num_workers=2)
    
    adaptor = Adaptor(
        channels=3, 
        num_classes=10, 
        hparams=hparams, 
        opt_name=args.opt_name,
        init_step=args.init_step,
        path_for_init=args.path_for_init
    )

    adaptor.to(device)

    train_minibatches_iterator = zip(train_loader)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = len(train_dataset)/hparams['batch_size']

    n_steps = args.steps
    checkpoint_freq = args.check_freq

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
        torch.save(save_dict, os.path.join(args.output_dir, filename))
    
    best_score = -float("inf")
    last_results_keys = None

    start_step = 0
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]

        # update
        step_vals = adaptor.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

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
            current_acc = misc.accuracy(adaptor, eval_loader, None, device)
            results['eval_acc'] = current_acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")


            if current_acc > best_score:
                best_score = current_acc
                print(f"Saving new best score at step: {step} at path: model_best.pkl")
                save_checkpoint(
                    'model_best.pkl',
                    results=json.dumps(results, sort_keys=True),
                )
                adaptor.to(device)

            algorithm_dict = adaptor.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')


    if args.init_step:
        adaptor.save_path_for_future_init(args.path_for_init)
        
    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')