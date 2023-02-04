## Weight Averaging for OOD Generalization and Few-shot Domain Adaptation

### How to run the code

#### In-Distribution Generalization on CIFAR100

**Step 1 Pre-train:** in this step, we will linear-probe ImageNet pretrained ResNet50 on the training split of CIFAR100.

```shell
python3 -m domainbed.scripts.train_id \
       --data_dir=../data \
       --output_dir=./CIFAR100_pretrain_sam_rho_0_05 \
       --algorithm SAM \
       --sam_rho 0.05 \
       --checkpoint_freq 100 \
       --init_step \
       --path_for_init ./CIFAR100_future_init_sam.pth
```

- `--algorithm` can be set to `SAM` and `ERM`, where `SAM` means **sharpness-aware minimization** and `ERM` means **empirical risk minimization**.
- `--init_step` is used to indicate only linear probing the final `fc` layer instead of fine-tuning the whole network.
- Pre-training will save a model into `path_for_init`, which will be used as future shared initialization in the next step.

**Step 2 Sweep train:** in this step, we utilize the pre-trained model in step 1 as shared initialization, and launch several independent runs to train with different hyper-parameters. Gradient similarity is used for training.

```shell
python3 -m domainbed.scripts.sweep_diverse_id launch \
        --data_dir=../data \
        --output_dir=./CIFAR100_sweep_grad_diverse_sam \
        --command_launcher local \
        --path_for_init ./CIFAR100_future_init_sam.pth \
        --algorithm ERM_2 \
        --sam_rho 0.05 \
        --n_hparams 20 \
        --n_trials 2 \
        --steps 20001 \
        --skip_confirmation
```

- `--algorithm` can be set to `ERM_2` or `SAM_2`, standing for **empirical risk minimization** and **sharpness-aware minimization**. The `_2` in the name is used to indicate this trained with gradient similarity. In the algorithm, at each time, 2 models will be trained together and the gradient similarity between these 2 models will be computed.
- `--n_hparams` used to set how many different hyper-parameter combinations we are going to sweep
- `--n_trials` used to set for each group of hyper-parameters, how many trials we are going to run.
- In total, we launch `20*2=40` models for this example, all the model will be saved in the `output_dir`.

**Step 3 Weight averaging: ** in this step, we will average the weights of these sweep trained models and test the averaged model on the test split of CIFAR100.

```shell
python3 -m domainbed.scripts.diwa_id \
       --data_dir=../data \
       --output_dir=./sxu/CIFAR100_sweep_grad_diverse_sam \
       --weight_selection uniform \
       --trial_seed -1
```

- `--weight_selection` used to indicate what types of weight averaging is used. In our implementation for gradient similarity, we only support `uniform` selection, which is averaging all the available independent models.



#### OOD Generalization on PACS/VLCS

**Step 1 Pre-train:** in this step, we will linear-probe a model to get a shared initialization for future sweep training. 

```shell
python3 -m domainbed.scripts.train \
       --data_dir=../data \
       --output_dir=./PACS_0_pretrain_sam_rho_0_05 \
       --algorithm SAM \
       --sam_rho 0.05 \
       --dataset PACS \
       --test_env 0 \
       --init_step \
       --path_for_init ./PACS_test_0_future_init_sam.pth \
       --steps 0
```

- `--algorithm` can be set to `SAM` or `ERM`, where `SAM` means **sharpness-aware minimization** and `ERM` means **empirical risk minimization**.

- `--dataset` used to set what dataset to be trained on, it can be `VLCS` and `PACS`.

- `--test_env` used to set which domain to be considered as out-of-distribution. For both `VLCS` and `PACS`, they have 4 domains, hence it can be set to `{0, 1, 2, 3}`. Here, the example shows we consider the `0-art` domain as the OOD data.

- Pre-training will save a model into `path_for_init`, which will be used as future shared initialization in the next step.

**Step 2 Sweep train:** in this step, we will utilize the pre-trained model in step 1 as a shared initialization, and launch several independent models training with different hyper-parameters. Gradient similarity is used for training.

```shell
python3 -m domainbed.scripts.sweep_diverse launch \
       --data_dir=../data \
       --output_dir=./PACS_0_sweep_grad_reg_sam_0_05 \
       --command_launcher local \
       --datasets PACS \
       --test_env 0 \
       --path_for_init ./PACS_test_0_future_init_sam.pth \
       --algorithms SAM_2 \
       --sam_rho 0.05 \
       --n_hparams 20 \
       --n_trials 2 \
       --skip_confirmation
```

- `--algorithm` can be set to `ERM_2` or `SAM_2`, standing for **empirical risk minimization** and **sharpness-aware minimization**. The `_2` in the name is used to indicate this trained with gradient similarity. In the algorithm, at each time, 2 models will be trained together and the gradient similarity between these 2 models will be computed.
- `--n_hparams` used to set how many different hyper-parameter combinations we are going to sweep
- `--n_trials` used to set for each group of hyper-parameters, how many trials we are going to run.
- In total, we launch `20*2=40` models for this example, all the model will be saved in the `output_dir`.

**Step 3 Weight averaging:** in this step, we will average the weights of these sweep trained models and test the averaged model on the out-of-distribution domain.

```shell
python3 -m domainbed.scripts.diwa_diverse \
       --data_dir=../data \
       --output_dir=./PACS_0_sweep_grad_reg_sam_0_05 \
       --dataset PACS \
       --test_env 0 \
       --weight_selection uniform \
       --num_models 15 \
       --num_trials 3 \
       --trial_seed -1
```

- `--num_models` is used to set how many models we are going to use for weight averaging. In this example we choose 15 models.
- `--num_trials` is used to set how many trials we are going to perform the weight averaging test. In this example we set it to 3, which means we are going to randomly select 15 models for 3 times, and averaged the 3 test accuracies for final report.



#### Few-shot Domain Adaptation: MNIST$\rightarrow$MNIST-M/USPS/SVHN

**Step 1 Pre-train:** in this step, we will linear probe a model on a source digits dataset.

```shell
python3 -m domainbed.scripts.few_shot_train \
    --data_dir=../data \
    --train_data MNIST \
    --num_classes 10 \
    --opt_name SAM \
    --model_name resnet18 \
    --model_pretrained \
    --output_dir=./mnist_res18_imagenet_sam_pretrain \
    --path_for_init ./mnist_res18_imagenet_future_init_sam.pth \
    --steps 8000 \
    --check_freq 800 \
    --linear_probe
```

- `--train_data` can be set to `MNIST`, `USPS`, `SVHN`, etc. depending on the specific adaptation task you are going to conduct.
- `--opt_name` can be set to `Adam` or `SAM`.
- `--model_name` can be set to `CNN`, `resnet18` or `resnet50`. When `CNN` is set, the model will be a simple 2-layer convolutional neural network.
- `--model_pretrained` only works for `resnet18` and `resnet50`. When it is set, the ImageNet pretrained model will be used.
- `--linear_probe` is set to use linear probing.

**Step 2 Sweep train:** in this step, we will launch several independent runs using the shared initialization pretrained in step 1.

```shell
python3 -m domainbed.scripts.sweep_few_shot launch \
    --data_dir=../data \
    --output_dir=./mnist_res18_imagenet_sweep_diwa_sam \
    --train_data MNIST \
    --num_classes 10 \
    --path_for_init ./mnist_res18_imagenet_future_init_sam.pth \
    --command_launcher local \
    --model_name resnet18 \
    --opt_name SAM \
    --steps 10000 \
    --check_freq 1000 \
    --n_hparams 10 \
    --n_trials 1 \
    --skip_confirmation
```

- The parameter meanings are similar to the above examples. Here, 10 individual models will be trained using different hyper-parameters.

**Step 3 Weight averaging and adaptation:** in this step, we will average the sweep trained models to obtain the averaged model. After that, the averaged model will be adapted on a few samples from the training split of target domain and then test on the test split of the target domain.

```shell
python3 -m domainbed.scripts.few_shot_adapt_after_WA \
    --data_dir=../data \
    --model_name resnet18 \
    --target_dataset MNISTM \
    --num_classes 10 \
    --sweep_dir=./mnist_res18_imagenet_sweep_diwa_sam \
    --output_dir=./mnist_res18_sam_adapt_2_mnistm_10_shot \
    --weight_selection uniform \
    --opt_name SAM \
    --sam_rho 0.05 \
    --k_shot 10 \
    --steps 2000 \
    --test_freq 10
```

- `--target_dataset` defines which dataset you are going to adapt to. It can be set to `MNISTM`, `USPS` and `SVHN`, depending on your specific task.

- This examples performs few-shot adaptation after weight averaging, which means it will do the weight averaging first and then fine-tune the averaged model on the target data.

- We also provide a code to perform few-shot adaptation before weight averaging, in which case, each individual model will be fine-tuned independently and then averaged to obtain the model.

- ```shell
  python3 -m domainbed.scripts.few_shot_adapt_before_WA \
      --data_dir=../data \
      --model_name resnet18 \
      --target_dataset MNISTM \
      --num_classes 10 \
      --sweep_dir=./mnist_res18_imagenet_sweep_diwa_sam \
      --weight_selection uniform \
      --opt_name SAM \
      --sam_rho 0.05 \
      --k_shot 10 \
      --steps 500 \
      --test_freq 10
  ```

- Experiments show that adaptation after weight averaging can achieve better performance.
