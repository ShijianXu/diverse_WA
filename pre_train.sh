python3 -m domainbed.scripts.train \
       --data_dir=../data \
       --algorithm ERM\
       --dataset VLCS \
       --test_env 1 \
       --init_step \
       --path_for_init ./VLCS_test_1_init.pth \
       --steps 0
