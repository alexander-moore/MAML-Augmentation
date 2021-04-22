# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torchvision

from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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


# def conv3x3(in_channels, out_channels, **kwargs):
#     return MetaSequential(
#         MetaConv2d(in_channels, out_channels, kernel_size=2, padding=1, **kwargs),
#         MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
#         nn.ReLU(),
#         nn.MaxPool2d(2)
#     )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features,num_classes, hidden_size=2):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size


        self.conv = MetaSequential(
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
            nn.Relu()
        )
        self.classifier = MetaLinear(hidden_size**10, num_classes)

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features')) ## can do inner loop and just upate cnn or ae
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
def get_data():
    np.random.seed(1)
    random.seed(1)

    data = np.load('data/RESISC45_images_96.npy')/255
    labels = np.load('data/RESISC45_classes.npy')

    test_size = 0.25
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=test_size, stratify=labels)

    np.save('data/RESISC45_images_train.npy', xtrain)
    np.save('data/RESISC45_labels_train.npy', ytrain)
    np.save('data/RESISC45_images_test.npy', xtest)
    np.save('data/RESISC45_labels_test.npy', ytest)
    train_data = np.load('data/RESISC45_images_train.npy')
    train_labels = np.load('data/RESISC45_labels_train.npy')
    classes = np.load('data/RESISC45_class_names.npy')
    img_size = train_data.shape[2]  # can use this to mofidy data size to fit this model (which only takes 256 images)
    bs = 32  # 64
    c_dim = classes.shape[0]
    xtrain, xval, ytrain, yval = train_test_split(train_data, train_labels, test_size=0.25)

    xtrain = torch.tensor(xtrain).permute(0, 3, 1, 2)
    print(torch.min(xtrain), torch.max(xtrain))

    trainset = []
    for i in range(xtrain.shape[0]):
        trainset.append((xtrain[i], ytrain[i]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                               shuffle=False)  # BUG: must keep shuffle false - or else it screws up labels, apparently

    ## Validation Data
    valset = []
    xval = torch.tensor(xval).permute(0, 3, 1, 2)

    print(torch.min(xval), torch.max(xval))
    for i in range(xval.shape[0]):
        valset.append((xval[i], yval[i]))

    val_loader = torch.utils.data.DataLoader(valset, batch_size=64, drop_last=True,
                                             shuffle=False)  # BUG: must keep shuffle false - or else it screws up labels, apparently

    test_data = np.load('data/RESISC45_images_test.npy')
    test_labels = np.load('data/RESISC45_labels_test.npy')

    test_data = torch.tensor(test_data)
    test_labels = torch.tensor(test_labels)
    ## Testing Data
    testset = []
    print(test_data.shape)
    xtest = torch.tensor(test_data).permute(0, 3, 1, 2)
    print(xtest.shape)

    print(torch.min(test_data), torch.max(test_data))
    for i in range(test_data.shape[0]):
        testset.append((test_data[i], test_labels[i]))

    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, drop_last=True,
                                              shuffle=False) # BUG: must keep shuffle false - or else it screws up labels, apparently
    Params= {img_size: "Img_size", bs: "bs", c_dim: "c_dim",}

    return train_loader,val_loader,test_loader, Params


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.zero_()


def one_hot_embedding(labels):
    labels = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64), num_classes=c_dim)
    return torch.squeeze(labels)


def top_k_acc(inp, targ, k):
    # print(inp.shape)
    tops = torch.topk(inp, k=k, dim=1)

    i = 0
    corrects = 0
    for row in tops:
        for element in row:
            if element == targ[i]:
                corrects += 1

        i += 1

    return corrects / inp.shape[0]


def accuracy_topk(output, target, topk=(3,)):
    # https://forums.fast.ai/t/return-top-k-accuracy/27658
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
