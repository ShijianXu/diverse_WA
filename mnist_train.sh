python3 -m domainbed.scripts.few_shot_mnist_train \
    --data_dir=../data \
    --opt_name SAM \
    --sam_rho 0.05 \
    --output_dir=./mnist_pretrain \
    --steps 10 \
    --check_freq 5
