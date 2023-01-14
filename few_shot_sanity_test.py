import torch
import torch.utils.data

from domainbed.lib import misc
from domainbed import few_shot_datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.few_shot_model import Adaptor

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hparams = {}
    hparams['lr'] = 5e-4
    hparams['rho'] = 0.05
    hparams['weight_decay'] = 5e-4
    hparams['batch_size'] = 8

    model = Adaptor(channels=3, 
        num_classes=10, 
        hparams=hparams, 
        opt_name='Adam', 
        model_name='CNN')
    
    model_path = './mnist_cnn_adam_adapt_2_svhn_10_shot/model.pkl'
    save_dict = torch.load(model_path, map_location=torch.device(device))
    state_dict = save_dict["model_dict"]
    new_state_dict = {key.replace("network_wa.", "classifier."): value for key, value in state_dict.items()}    

    missing_keys, unexpected_keys =  model.load_state_dict(new_state_dict, strict=False)
    print(f"Load individual model with missing keys {missing_keys} and unexpected keys {unexpected_keys}.")

    test_dataset  = few_shot_datasets.get_dataset("../data", 'SVHN', 64, False)
    eval_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=64,
        num_workers=2)

    model.to(device=device)
    model.eval()
    acc = misc.accuracy(model, eval_loader, None, device)
    print(acc)