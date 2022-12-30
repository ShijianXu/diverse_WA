import numpy as np
from domainbed.lib import misc


def _hparams(opt_name, random_seed, data_name):
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))
        print(f"rand hparam {name} with value {hparams[name]}")

    if data_name == 'VisDA':
        _hparam('lr', 5e-4, lambda r: r.choice([5e-4, 1e-5, 5e-5]))
    else:
        _hparam('lr', 5e-4, lambda r: r.choice([1e-4, 3e-4, 5e-4]))
        
    _hparam('weight_decay', 0, lambda r: r.choice([1e-4, 1e-6]))
    _hparam('batch_size', 32, lambda r: 32)

    if opt_name == 'SAM':
        _hparam('rho', 0.05, lambda r: r.choice([0.01, 0.02, 0.05, 0.1]))

    return hparams

def default_hparams(opt_name, data_name='MNIST'):
    return {a: b for a, (b, c) in _hparams(opt_name, 0, data_name).items()}


def random_hparams(opt_name, seed, data_name='MNIST'):
    return {a: c for a, (b, c) in _hparams(opt_name, seed, data_name).items()}