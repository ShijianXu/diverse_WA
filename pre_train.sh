python3 -m domainbed.scripts.train \
       --data_dir=../data \
       --algorithm ERM\
       --dataset PACS\
       --test_env 0 \
       --init_step \
       --path_for_init ./pretrain.pth \
       --steps 0
