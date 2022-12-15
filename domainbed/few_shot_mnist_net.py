import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

from domainbed.sam import SAMin
from torch.nn.modules.batchnorm import _BatchNorm

class Classifier(nn.Module):
    def __init__(self, channels=3, num_classes=10):
        super(Classifier, self).__init__()
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
    def __init__(self, channels, num_classes, hparams, opt_name='Adam'):
        super(Adaptor, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.classifier = Classifier(self.channels, self.num_classes)
        self.opt_name = opt_name

        if self.opt_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.opt_name == 'SAM':
            base_optimizer = torch.optim.Adam
            self.optimizer = SAMin(
                self.classifier.parameters(),
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

        self.network.apply(_enable)

    def disable_running_stats(self):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        self.network.apply(_disable)


if __name__ == '__main__':
    net = Classifier()
    x = torch.rand(2, 3, 64, 64)
    out = net(x)
    print(out.shape)