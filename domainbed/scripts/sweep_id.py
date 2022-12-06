# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

"""
python3 -m domainbed.scripts.sweep_id launch \
       --data_dir=../data \
       --output_dir=./CIFAR100_sweep_diwa_sam_rho_0_01 \
       --command_launcher local \
       --path_for_init ./CIFAR100_init_sam_rho_0.01.pth \
       --algorithms SAM \
       --sam_rho 0.01 \
       --n_hparams 10 \
       --n_trials 1 \
       --steps 20001 \
       --skip_confirmation
"""

import argparse
import copy
import hashlib
import json
import os
import shutil

import numpy as np

from domainbed import hparams_registry
from domainbed import algorithms
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
        command = ['python', '-m', 'domainbed.scripts.train_id']
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
        job_info = (
            self.train_args['algorithm'],
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

def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials, algorithms, n_hparams_from, n_hparams, steps,
    data_dir, task, holdout_fraction, hparams, path_for_init):
    args_list = []
    for trial_seed in range(n_trials):
        for algorithm in algorithms:
            for hparams_seed in range(n_hparams_from, n_hparams):
                train_args = {}
                train_args['algorithm'] = algorithm
                train_args['holdout_fraction'] = holdout_fraction
                train_args['hparams_seed'] = hparams_seed
                train_args['data_dir'] = data_dir
                train_args['task'] = task
                train_args['trial_seed'] = trial_seed
                train_args['path_for_init'] = path_for_init
                train_args['seed'] = misc.seed_hash(algorithm, hparams_seed, trial_seed)
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
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_confirmation', action='store_true')
     ## DiWA ##
    parser.add_argument('--path_for_init', type=str, default=None)
    parser.add_argument('--sam_rho', type=float, default=0.05)
    
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        hparams=args.hparams,
        ## DiWA ##
        path_for_init=args.path_for_init
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
