"""
# train baseline
python3 -m domainbed.scripts.few_shot_train \
    --data_dir=../data \
    --train_data MNIST \
    --num_classes 10 \
    --opt_name Adam \
    --model_name resnet18 \
    --output_dir=./mnist_res18_pretrain \
    --path_for_init ./mnist_res18_future_init_adam.pth \
    --steps 500 \
    --check_freq 5 \
    --linear_probe

python3 -m domainbed.scripts.few_shot_train \
    --data_dir=../data \
    --train_data VisDA \
    --num_classes 12 \
    --opt_name SAM \
    --model_name resnet50 \
    --model_pretrained \
    --linear_probe \
    --output_dir=./res50_visda_lineprobe_sam \
    --path_for_init ./res50_visda_lineprobe_sam_future_init.pth \
    --steps 5000 \
    --check_freq 1000
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

from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed import hparams_registry_few_shot
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.few_shot_model import Adaptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptation MNIST Pretrain")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--opt_name', type=str, default="Adam")
    # parser.add_argument('--train', action='store_true')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--check_freq', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="few_shot_pretrain")
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--path_for_init', type=str, default=None)
    parser.add_argument('--linear_probe', action='store_true')
    # parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--model_name', type=str, default="resnet18")
    parser.add_argument('--model_pretrained', action='store_true')
    parser.add_argument('--train_data', type=str, default='MNIST')
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # set hyperparameters
    if args.hparams_seed == 0:
        hparams = hparams_registry_few_shot.default_hparams(args.opt_name, args.train_data)
    else:
        hparams = hparams_registry_few_shot.random_hparams(args.opt_name, misc.seed_hash(args.hparams_seed, args.trial_seed), args.train_data)

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

    if args.train_data == 'VisDA':
        visda_dataset = few_shot_datasets.get_dataset(data_dir, args.train_data, 64, True, k_shot=-1)
        train_dataset = visda_dataset.train_dataset
        test_dataset = visda_dataset.test_datas
    else:
        train_dataset = few_shot_datasets.get_dataset(data_dir, args.train_data, 64, True, k_shot=-1)
        test_dataset  = few_shot_datasets.get_dataset(data_dir, args.train_data, 64, False, k_shot=-1)

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=None,
        batch_size=hparams['batch_size'],
        num_workers=1)

    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=hparams['batch_size'],
        num_workers=1)
    
    adaptor = Adaptor(
        channels=3, 
        num_classes=args.num_classes, 
        hparams=hparams, 
        opt_name=args.opt_name,
        model_name=args.model_name,
        model_pretrained=args.model_pretrained,
        linear_probe=args.linear_probe,
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


    if args.linear_probe:
        print("Saving model for future init.")
        adaptor.save_path_for_future_init(args.path_for_init)
        
    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')