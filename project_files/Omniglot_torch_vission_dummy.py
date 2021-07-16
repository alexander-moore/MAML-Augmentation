# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torchvision

from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy
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
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
def train(model,dataloader,num_batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    acc= []
    with tqdm(dataloader, total=num_batch) as pbar:
            for batch_idx, batch in enumerate(pbar):
                model.zero_grad()

                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=device)
                train_targets = train_targets.to(device=device)

                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=device)
                test_targets = test_targets.to(device=device)

                outer_loss = torch.tensor(0., device=device)
                accuracy = torch.tensor(0., device=device)
                for task_idx, (train_input, train_target, test_input,
                        test_target) in enumerate(zip(train_inputs, train_targets,
                        test_inputs, test_targets)):
                    train_logit = model(train_input)
                    inner_loss = F.cross_entropy(train_logit, train_target)

                    model.zero_grad()
                    params = gradient_update_parameters(model,
                                                        inner_loss,
                                                        step_size=.005,
                                                        first_order=True)

                    test_logit = model(test_input, params=params)
                    outer_loss += F.cross_entropy(test_logit, test_target)

                    with torch.no_grad():
                        accuracy += get_accuracy(test_logit, test_target)

                outer_loss.div_(16)
                accuracy.div_(16)

                outer_loss.backward()
                meta_optimizer.step()
                acc.append(accuracy.to("cpu"))
                # plt.clf()
                # plt.plot(np.asarray(acc))
                # plt.ylabel('acc')
                # plt.show()
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
                if batch_idx >= num_batch:
                    break
    return acc


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    dataset = Omniglot("data",
                       # Number of ways
                       num_classes_per_task=5,
                       # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                       transform=Compose([Resize(28), ToTensor()]),
                       # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                       target_transform=Categorical(num_classes=5),
                       # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                       class_augmentations=[Rotation([90, 180, 270])],
                       meta_train=True,
                       download=True)
    # split the data into train and test
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=5, num_test_per_class=15)
    # creating batches from dataset
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=1)# (16, 75)
    # import the required libraries and Meta modules from torchmeta
    import torch.nn as nn
    from torchmeta.modules import (MetaModule, MetaSequential,
                                   MetaConv2d, MetaLinear)

    model = ConvolutionalNeuralNetwork(1,
                                       5,##num ways
                                       hidden_size=64)


    acccc = train(model,dataloader,1000)
    plt.plot(np.asarray(acccc))
    plt.ylabel('acc')
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
