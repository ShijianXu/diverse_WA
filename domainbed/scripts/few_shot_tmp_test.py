
from domainbed import few_shot_datasets

dataset = 'PACS'
data_dir = '../data'
test_env = 1

hparams = {}
hparams['data_augmentation'] = True

if dataset in vars(few_shot_datasets):
    dataset = vars(few_shot_datasets)[dataset](data_dir, 1, 10)

print(len(dataset.train_dataset))
print(len(dataset.test_dataset))

