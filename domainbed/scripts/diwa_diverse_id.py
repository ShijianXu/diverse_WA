"""
python3 -m domainbed.scripts.diwa_diverse_id \
       --data_dir=../data \
       --output_dir=/scratch/izar/sxu/CIFAR100_sam_rho_0_01_sweep_erm_grad \
       --weight_selection uniform \
       --trial_seed -1
"""


import argparse
import os
import json
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import transforms

from domainbed import datasets, algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")

    # select which checkpoints
    parser.add_argument('--weight_selection', type=str, default="uniform") # or "restricted"
    parser.add_argument(
        '--trial_seed',
        type=int,
        default="-1",
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )

    inf_args = parser.parse_args()
    return inf_args


def get_dict_folder_to_score(inf_args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    output_folders = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    output_folders = [
        output_folder for output_folder in output_folders
        if os.path.isdir(output_folder) and "done" in os.listdir(output_folder) and "model.pkl" in os.listdir(output_folder)
    ]

    dict_folder_to_score = {}
    for folder in output_folders:
        # model_path = os.path.join(folder, "model_best.pkl")
        # save_dict = torch.load(model_path, map_location=torch.device(device))
    
        # results = json.loads(save_dict["results"])
        # score = results["eval_acc"]

        score = 0

        print(f"Found: {folder} with score: {score}")
        dict_folder_to_score[folder] = score

    if len(dict_folder_to_score) == 0:
        raise ValueError(f"No folders found for: {inf_args}")
    return dict_folder_to_score


def get_wa_results(good_checkpoints, dataset, device):
    print(good_checkpoints)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    INPUT_SHAPE = (3, 224, 224)
    N_CLASSES = 100
    N_DOMAINS = 1
    wa_algorithm = algorithms_inference.DiWA(
        INPUT_SHAPE,
        N_CLASSES,
        N_DOMAINS,
    )
    for folder in good_checkpoints:
        save_dict = torch.load(os.path.join(folder, "model.pkl"), map_location=torch.device(device))
        train_args = save_dict["args"]

        # load individual weights
        algorithm = algorithms_inference.ERM_2(
            INPUT_SHAPE, 
            N_CLASSES,
            N_DOMAINS,
            save_dict["model_hparams1"],
            save_dict["model_hparams2"],
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        
        wa_algorithm.add_weights(algorithm.network1)
        wa_algorithm.add_weights(algorithm.network2)
        del algorithm

    wa_algorithm.to(device)
    wa_algorithm.eval()
    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader = FastDataLoader(
            dataset=dataset,
            batch_size=64,
            num_workers=4)

    dict_results = {}
    dict_results["acc"] = misc.accuracy(wa_algorithm, loader, None, device)
    dict_results["length"] = len(good_checkpoints)
    return dict_results



def print_results(dict_results):
    results_keys = sorted(list(dict_results.keys()))
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([dict_results[key] for key in results_keys], colwidth=12)


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Begin DiWA for: {inf_args} with device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=test_transform)

    # load individual folders and their corresponding scores on train_out
    dict_folder_to_score = get_dict_folder_to_score(inf_args)

    # load data: test and optionally train_out for restricted weight selection
    data_splits, data_names = [], []
    dict_domain_to_filter = {"test": "full"}
    if inf_args.weight_selection == "restricted":
        assert inf_args.trial_seed != -1
        dict_domain_to_filter["train"] = "out"

    # compute score after weight averaging
    if inf_args.weight_selection == "restricted":
        # Restricted weight selection

        ## sort individual members by decreasing accuracy on train_out
        sorted_checkpoints = sorted(dict_folder_to_score.keys(), key=lambda x: dict_folder_to_score[x], reverse=True)
        selected_indexes = []
        best_result = -float("inf")
        dict_best_results = {}
        ## incrementally add them to the WA
        for i in range(0, len(sorted_checkpoints)):
            selected_indexes.append(i)
            selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

            ood_results = get_wa_results(
                selected_checkpoints, dataset, data_names, data_splits, device
            )
            ood_results["i"] = i
            ## accept only if WA's accuracy is improved
            if ood_results["train_acc"] >= best_result:
                dict_best_results = ood_results
                ood_results["accept"] = 1
                best_result = ood_results["train_acc"]
                print(f"Accepting index {i}")
            else:
                ood_results["accept"] = 0
                selected_indexes.pop(-1)
                print(f"Skipping index {i}")
            print_results(ood_results)

        ## print final scores
        dict_best_results["final"] = 1
        print_results(dict_best_results)

    elif inf_args.weight_selection == "uniform":
        dict_results = get_wa_results(
            list(dict_folder_to_score.keys()), dataset, device
        )
        print_results(dict_results)

    else:
        raise ValueError(inf_args.weight_selection)


if __name__ == "__main__":
    main()
