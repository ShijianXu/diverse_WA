import torch
import torch.utils.data

from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.few_shot_model import DiWA_Adaptor

if __name__ == '__main__':
    hparams = {}
    hparams['lr'] = 5e-4
    hparams['rho'] = 0.05
    hparams['weight_decay'] = 5e-4
    hparams['batch_size'] = 8

    model = DiWA_Adaptor(channels=3, num_classes=10, hparams=hparams, opt_name='Adam')
    model_path = './mnist_cnn_adam_adapt_2_svhn_10_shot/model.pkl'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dict = torch.load(model_path, map_location=torch.device(device))
    missing_keys, unexpected_keys =  model.load_state_dict(save_dict["model_dict"], strict=False)
    print(f"Load individual model with missing keys {missing_keys} and unexpected keys {unexpected_keys}.")

    test_dataset  = few_shot_datasets.get_dataset("../data", 'SVHN', 64, False)
    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=2)

    model.eval()
    acc = misc.accuracy(model, eval_loader, None, device)
    print(acc)