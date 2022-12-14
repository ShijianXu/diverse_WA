python3 -m domainbed.scripts.sweep_mnist launch \
    --data_dir=../data \
    --output_dir=./mnist_sweep_diwa_adam \
    --path_for_init ./mnist_future_init_adam.pth \
    --command_launcher local \
    --opt_name Adam \
    --steps 10000 \
    --check_freq 1000 \
    --n_hparams 10 \
    --n_trials 1 \
    --skip_confirmation
