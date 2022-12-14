"""
# Diverse weight averaging training for MNIST
python3 -m domainbed.scripts.sweep_few_shot launch \
    --data_dir=../data \
    --train_data MNIST \
    --num_classes 10 \
    --output_dir=./MNIST_sweep_diwa_adam \
    --path_for_init ./mnist_cnn_future_init_adam.pth \
    --command_launcher local \
    --opt_name Adam \
    --model_name CNN \
    --steps 10000 \
    --check_freq 1000 \
    --n_hparams 10 \
    --n_trials 1 \
    --skip_confirmation


python3 -m domainbed.scripts.sweep_few_shot launch \
    --data_dir=../data \
    --train_data VisDA \
    --num_classes 12 \
    --output_dir=./VisDA_sweep_diwa_sam \
    --path_for_init ./res50_visda_lineprobe_sam_future_init.pth \
    --command_launcher local \
    --opt_name SAM \
    --model_name resnet50 \
    --steps 10000 \
    --check_freq 1000 \
    --n_hparams 10 \
    --n_trials 1 \
    --skip_confirmation
"""

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex


class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'domainbed.scripts.few_shot_train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['opt_name'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def make_args_list(n_trials, opt_name, n_hparams_from, n_hparams, steps, data_dir, train_data, num_classes, hparams, path_for_init, model_name, check_freq):
    args_list = []
    for trial_seed in range(n_trials):
        for hparams_seed in range(n_hparams_from, n_hparams):
            train_args = {}
            train_args['opt_name'] = opt_name
            train_args['hparams_seed'] = hparams_seed
            train_args['data_dir'] = data_dir
            train_args['train_data'] = train_data
            train_args['num_classes'] = num_classes
            train_args['trial_seed'] = trial_seed
            train_args['path_for_init'] = path_for_init
            train_args['model_name'] = model_name
            train_args['seed'] = misc.seed_hash(opt_name, hparams_seed, trial_seed)
            train_args['check_freq'] = check_freq

            if steps is not None:
                train_args['steps'] = steps
            if hparams is not None:
                train_args['hparams'] = hparams

            args_list.append(train_args)

    return args_list


def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DiWA sweep for MNIST')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train_data', type=str, default='MNIST')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--opt_name', type=str, default="Adam")
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--check_freq', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="MNIST_sweep")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=10)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--path_for_init', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="resnet18")

    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        opt_name=args.opt_name,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        train_data=args.train_data,
        num_classes=args.num_classes,
        hparams=args.hparams,
        path_for_init=args.path_for_init,
        model_name=args.model_name,
        check_freq=args.check_freq
    )

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)