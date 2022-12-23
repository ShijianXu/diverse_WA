python3 -m domainbed.scripts.few_shot_adapt_before_WA \
    --data_dir=../data \
    --target_dataset SVHN \
    --sweep_dir=./mnist_res18_sweep_diwa_sam \
    --weight_selection uniform \
    --opt_name SAM \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 100
