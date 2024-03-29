python3 -m domainbed.scripts.few_shot_adapt_after_WA \
    --data_dir=../data \
    --model_name resnet18 \
    --target_dataset MNIST \
    --num_classes 10 \
    --sweep_dir=./usps_res18_imagenet_sweep_diwa_sam \
    --output_dir=./usps_res18_imagenet_sam_adapt_2_mnist_10_shot \
    --weight_selection uniform \
    --opt_name SAM \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 300 \
    --test_freq 10
