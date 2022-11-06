python3 -m domainbed.scripts.train \
       --data_dir=../data \
       --output_dir=./VLCS_baseline \
       --algorithm ERM\
       --dataset VLCS \
       --test_env 1 \
       --init_step \
       --path_for_init ./VLCS_test_1_init.pth \
       --steps 0
