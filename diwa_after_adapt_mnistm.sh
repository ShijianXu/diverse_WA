python3 -m domainbed.scripts.few_shot_adapt_mnistm_before_WA \
    --data_dir=../data \
    --target_dataset SVHN \
    --sweep_dir=./mnist_sweep_diwa_adam \
    --output_dir=./mnist_adam_diwa_2_svhn_adam_10_shot \
    --weight_selection uniform \
    --opt_name Adam \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 100
