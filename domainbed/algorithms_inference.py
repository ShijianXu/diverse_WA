
import copy
import torch.nn as nn
from domainbed import networks, algorithms

class ERM(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier']
        )
        self.num_classes = num_classes
        self.network = nn.Sequential(self.featurizer, self.classifier)


class DiWA(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams={})
        self.network_wa = None
        self.global_count = 0

    def add_weights(self, network):
        if self.network_wa is None:
            self.network_wa = copy.deepcopy(network)
        else:
            for param_q, param_k in zip(network.parameters(), self.network_wa.parameters()):
                param_k.data = (param_k.data * self.global_count + param_q.data) / (1. + self.global_count)
        self.global_count += 1

    def predict(self, x):
        return self.network_wa(x)


#===============================
class ERM_2(nn.Module):

    def __init__(self, input_shape, num_classes, num_domains, hparams1, hparams2):
        super(ERM_2, self).__init__()

        self.hparams1 = hparams1
        self.hparams2 = hparams2

        # model_1
        self.featurizer1 = networks.Featurizer(input_shape, self.hparams1)
        self.classifier1 = networks.Classifier(
            self.featurizer1.n_outputs,
            num_classes,
            self.hparams1['nonlinear_classifier'])
        self.network1 = nn.Sequential(self.featurizer1, self.classifier1)

        # model_2
        self.featurizer2 = networks.Featurizer(input_shape, self.hparams2)
        self.classifier2 = networks.Classifier(
            self.featurizer2.n_outputs,
            num_classes,
            self.hparams2['nonlinear_classifier'])
        self.network2 = nn.Sequential(self.featurizer2, self.classifier2)
