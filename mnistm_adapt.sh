python3 -m domainbed.scripts.few_shot_adapt_mnistm \
    --data_dir=../data \
    --opt_name SAM \
    --output_dir=./mnist_adam_2_mnistm_sam_10_shot \
    --model_path ./mnist_pretrain_Adam/model_best.pkl \
    --k_shot 10 \
    --steps 110
