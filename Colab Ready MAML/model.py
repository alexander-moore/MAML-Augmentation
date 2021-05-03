import torch.nn as nn
import utils
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self,Params,in_channels=3,out_features=45,hidden_size=2):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = Params['in_channels']
        self.out_features = Params['num_classes']
        self.hidden_size = Params['hidden_size']

        self.features  = MetaSequential(
            MetaConv2d(in_channels,hidden_size**5,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            MetaConv2d(hidden_size**5,hidden_size**6,kernel_size=4,stride=2,padding=1,bias=False),
            MetaBatchNorm2d(hidden_size**6),
            nn.ReLU(),
            MetaConv2d(hidden_size ** 6, hidden_size ** 7, kernel_size=4, stride=2, padding=1, bias=False),
            MetaBatchNorm2d(hidden_size**7),
            nn.ReLU(),
            MetaConv2d(hidden_size ** 7, hidden_size ** 8, kernel_size=4, stride=2, padding=1, bias=False),
            MetaBatchNorm2d(hidden_size ** 8),
            nn.ReLU(),
            MetaConv2d(hidden_size ** 8, hidden_size ** 9, kernel_size=4, stride=2, padding=1, bias=False),
            MetaBatchNorm2d(hidden_size ** 9),
            nn.ReLU(),
            MetaConv2d(hidden_size ** 9, hidden_size ** 10, kernel_size=4, stride=2, padding=1, bias=False),
            MetaBatchNorm2d(hidden_size ** 10),
            nn.ReLU(),
        )
        self.classifier = MetaLinear(hidden_size**10, out_features)



    def weight_init(self):
        for m in self.features:
            utils.normal_init(self._modules[m])
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


