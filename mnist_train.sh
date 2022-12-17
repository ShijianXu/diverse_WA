python3 -m domainbed.scripts.few_shot_mnist_train \
    --data_dir=../data \
    --opt_name SAM \
    --output_dir=./mnist_init_train_sam \
    --path_for_init ./mnist_future_init_sam.pth \
    --steps 100 \
    --check_freq 100 \
    --init_step
