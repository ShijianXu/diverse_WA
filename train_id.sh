python3 -m domainbed.scripts.train_id \
       --data_dir=../data \
       --output_dir=./CIFAR100_pretrain \
       --algorithm ERM \
       --checkpoint_freq 100 \
       --init_step \
       --path_for_init ./CIFAR100_init.pth \
       --steps 0
