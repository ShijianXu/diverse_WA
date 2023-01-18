python3 -m domainbed.scripts.train_id \
       --data_dir=../data \
       --output_dir=./CIFAR100_pretrain_test \
       --algorithm SAM \
       --sam_rho 0.05 \
       --checkpoint_freq 100 \
       --init_step \
       --path_for_init /scratch/izar/sxu/CIFAR100_init_sam_test.pth