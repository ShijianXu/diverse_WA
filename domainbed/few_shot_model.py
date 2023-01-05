import os

import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

import torchvision

from domainbed.sam import SAMin
from torch.nn.modules.batchnorm import _BatchNorm

import copy


class CNN(nn.Module):
    def __init__(self, channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(12288, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class Adaptor(torch.nn.Module):
    def __init__(self, channels, num_classes, hparams, opt_name='Adam', model_name='resnet18', model_pretrained=False, linear_probe=False, path_for_init=None):
        super(Adaptor, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        
        if model_name == 'CNN':
            self.classifier = CNN(self.channels, self.num_classes)
        elif model_name == 'resnet18':
            if model_pretrained:
                self.classifier = torchvision.models.resnet18(pretrained=True)
                num_ftrs = self.classifier.fc.in_features
                self.classifier.fc = nn.Linear(num_ftrs, num_classes)
                print("Define ResNet18 with pretrained weights.")
            else:
                self.classifier = torchvision.models.resnet18(num_classes=num_classes)
                print("Define ResNet18 without pretrained weights.")
        elif model_name == 'resnet50':
            if model_pretrained:
                self.classifier = torchvision.models.resnet50(pretrained=True)
                num_ftrs = self.classifier.fc.in_features
                self.classifier.fc = nn.Linear(num_ftrs, num_classes)
                print("Define ResNet50 with pretrained weights.")
            else:
                self.classifier = torchvision.models.resnet50(num_classes=num_classes)
                print("Define ResNet50 without pretrained weights.")

        if path_for_init is not None:
            if os.path.exists(path_for_init):
                self.classifier.load_state_dict(torch.load(path_for_init))
                print(f"Loaded linear probed shared init model {path_for_init}.")
            else:
                assert linear_probe, "Your initialization has not been saved yet"

        self.hparams = hparams
        self.opt_name = opt_name

        if linear_probe and model_pretrained:
            # linear probing
            parameters_to_be_optimized = self.classifier.fc.parameters()
            print("Only linear probe the fc layer.")
        else:
            # train from scratch or fine-tuning the whole network
            parameters_to_be_optimized = self.classifier.parameters()
            print("Tuning the whole network.")

        if self.opt_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                #self.classifier.parameters(),
                parameters_to_be_optimized,
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.opt_name == 'SAM':
            base_optimizer = torch.optim.Adam
            self.optimizer = SAMin(
                #self.classifier.parameters(),
                parameters_to_be_optimized,
                base_optimizer,
                rho=self.hparams["rho"],
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

    def update(self, minibatches):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        if self.opt_name == 'Adam':
            # training loss
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        elif self.opt_name == "SAM":
            # first forward-backward pass
            self.enable_running_stats()
            loss = F.cross_entropy(self.predict(all_x), all_y)      # use this loss for any training statistics
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            self.disable_running_stats()
            F.cross_entropy(self.predict(all_x), all_y).backward()  # make sure to do a full forward pass
            self.optimizer.second_step(zero_grad=True)

        else:
            raise NotImplementedError

        return {'loss': loss.item()}

    def predict(self, x):
        return self.classifier(x)

    def enable_running_stats(self):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        self.classifier.apply(_enable)

    def disable_running_stats(self):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        self.classifier.apply(_disable)

    def save_path_for_future_init(self, path_for_init):
        assert not os.path.exists(path_for_init), "The initialization has already been saved"
        torch.save(self.classifier.state_dict(), path_for_init)


class DiWA_Adaptor(torch.nn.Module):
    def __init__(self, channels, num_classes, hparams, opt_name='Adam'):
        super().__init__()

        self.hparams = hparams

        self.network_wa = None
        self.global_count = 0
        self.opt_name = opt_name

    def add_weights(self, network):
        if self.network_wa is None:
            self.network_wa = copy.deepcopy(network)
        else:
            for param_q, param_k in zip(network.parameters(), self.network_wa.parameters()):
                param_k.data = (param_k.data * self.global_count + param_q.data) / (1. + self.global_count)
        self.global_count += 1

    def set_optimizer(self):
        if self.opt_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.network_wa.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.opt_name == 'SAM':
            base_optimizer = torch.optim.Adam
            self.optimizer = SAMin(
                self.network_wa.parameters(),
                base_optimizer,
                rho=self.hparams["rho"],
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

    def update(self, minibatches):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        if self.opt_name == 'Adam':
            # training loss
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        elif self.opt_name == "SAM":
            # first forward-backward pass
            self.enable_running_stats()
            loss = F.cross_entropy(self.predict(all_x), all_y)      # use this loss for any training statistics
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            self.disable_running_stats()
            F.cross_entropy(self.predict(all_x), all_y).backward()  # make sure to do a full forward pass
            self.optimizer.second_step(zero_grad=True)

        else:
            raise NotImplementedError

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network_wa(x)

    def enable_running_stats(self):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        self.network_wa.apply(_enable)

    def disable_running_stats(self):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        self.network_wa.apply(_disable)


if __name__ == '__main__':
    net = torchvision.models.resnet18(num_classes=10)
    x = torch.rand(2, 3, 64, 64)
    out = net(x)
    print(out.shape)