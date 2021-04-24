
## in your env for this project be sure to do the following installs
## pip install learn2learn
## pip install torchmeta
import augmentation_functions
import random
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

from torchmeta.utils.gradient_based import gradient_update_parameters

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
import learn2learn as l2l

##GLOBALS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##
class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self,Params,in_channels=3,num_classes=45,hidden_size=2):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = Params['in_channels']
        self.num_classes = Params['num_classes']
        self.hidden_size = Params['hidden_size']

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
def train(model,train_tasks,Params,val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=Params['MetaLR'])
    acc= []
    for _ in range(Params['epoch']):
        outer_loss = torch.tensor(0., device=device)
        accuracy = torch.tensor(0., device='cpu')
        ## run through all tasks
        for task,i in zip(train_tasks,range(Params['number_of_tasks'])):
            model.zero_grad()
            ## call augments on the fly
            task[0] = augmentation_functions.applyAugs(task[0], i) ## only apply augs to x
            x_inner,x_outer,y_inner,y_outer = tasksplit(task) ## split task for inner/ ouyter with same augs
            x_inner = x_inner.to(device)
            y_inner = y_inner.to(device)
            x_outer = x_outer.to(device)
            y_outer = y_outer.to(device)
            train_logit = model(x_inner)
            inner_loss = F.cross_entropy(train_logit, y_inner)
            model.zero_grad()
            params = gradient_update_parameters(model,
                                                inner_loss,
                                                #params=None, ## note we can do a for loop and take multiple inner loop steps, set to none since
                                                step_size=Params["innerStep"],
                                                first_order=Params["Order"])
            test_logit = model(x_outer, params=params) ## take the loss fucntions using the params of this task specific inner loop
            outer_loss += F.cross_entropy(test_logit, y_outer)

            with torch.no_grad():
                ## get other accuracy fucntiopns here
                accuracy += get_accuracy(test_logit, y_outer).to('cpu')

            outer_loss.div_(Params['nways']*Params['kshots'])
            accuracy.div_(Params['nways']*Params['kshots'])
            ## be sure to normalize any accuracy fucntions by batch size

        ## close out outer loops
        outer_loss.backward()
        meta_optimizer.step()
        acc.append(accuracy.to("cpu"))
        """
        needs to be integrated with Params #%
        it does not work as of now. train needs to integrate other acc's as well
        """


        """
        with torch.no_grad():
            accs, actk = [], []
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).float()
                # print(x, y)
                yhat = CNN(x)

                yhat_max = torch.max(yhat, dim=1)[1]
                # print(yhat.shape)

                correct = torch.sum(yhat_max == y)
                size = x.shape[0]

                acc_topk = accuracy_topk(yhat, y)
                # print(acc_topk)
                actk.append(acc_topk.data.item())

                accs.append(100 * (correct / size).data.item())
            print()
            print('Validation Accuracy: ', torch.mean(torch.FloatTensor(accs)).data.item())
            print('Validation Top3 Accuracy: ', torch.mean(torch.FloatTensor(actk)).data.item())
            print()

        val_accs.append(torch.mean(torch.FloatTensor(accs)))
        val_topks.append(torch.mean(torch.FloatTensor(actk)))
        """
    return acc
def linkDataset(x,y):
    set = []
    for i in range(len(y)):
        set.append((x[i], y[i]))
    return set


def taskStructure(dataset,Params):
    labels= {}
    for i in range(len(dataset.targets)):
        labels[i] = dataset.targets[i]
    metaDataset = l2l.data.MetaDataset(dataset,indices_to_labels=labels)
    transforms = [
        ## need kshots*2 for inner and outer loop split accross each task
        l2l.data.transforms.FusedNWaysKShots(metaDataset,n=Params['nways'],k=Params['kshot']*2,replacement=False),
        l2l.data.transforms.LoadData(metaDataset),
        l2l.data.transforms.ConsecutiveLabels(metaDataset)
    ]
    taskset = l2l.data.TaskDataset(metaDataset,
                                   transforms,
                                   num_tasks=math.floor((45*700)/(Params['nways']*Params['kshot']))
                                   ) ## generate max number of tasks
    return taskset

def tasksplit(task):
    ## split inner and outer loop by incidies
    x_inner = task[0][::2]
    y_inner = task[1][::2]
    x_outer = task[0][1::2]
    y_outer = task[1][1::2]
    return x_inner, x_outer, y_inner,y_outer
def get_data(Params):
    np.random.seed(1)
    random.seed(1)
    file = 'data/RESISC45_images_'+str(Params['datasize'])+'.npy'
    data = np.load(file)/256
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
    c_dim = classes.shape[0]
    xtrain, xval, ytrain, yval = train_test_split(train_data, train_labels, test_size=0.25)

    xtrain = torch.tensor(xtrain).permute(0, 3, 1, 2)
    trainset = linkDataset(xtrain,ytrain)

    train_tasks = taskStructure(trainset,Params)
    bs =Params['nways']*Params['kshot']
    ## Validation Data
    valset = []
    xval = torch.tensor(xval).permute(0, 3, 1, 2)
    valset = linkDataset(xval,yval)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=512, drop_last=True,
                                             shuffle=False)  # BUG: must keep shuffle false - or else it screws up labels, apparently

    test_data = np.load('data/RESISC45_images_test.npy')
    test_labels = np.load('data/RESISC45_labels_test.npy')

    test_data = torch.tensor(test_data)
    test_labels = torch.tensor(test_labels)
    ## Testing Data
    xtest = torch.tensor(test_data).permute(0, 3, 1, 2)
    testset = linkDataset(xtest,ytest)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=512, drop_last=True,
                                              shuffle=False) # BUG: must keep shuffle false - or else it screws up labels, apparently
    Params['Img_size']=img_size
    Params['bs'] = bs
    Params["c_dim"] = c_dim

    return train_tasks,val_loader,test_loader, Params


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
    Params = {'datasize':96,'nways':5,'kshots':1,'epoch':10,'in_channels':3,
              'num_classes':45,'hidden_size':2,"datasize" : 96, "innerstep":.05,
              'order':False,'MetaLR':1e-3,"number_of_tasks":128}
    ## need to add an eval every epoch on train
    train_tasks,val_loader,test_loader=get_data(Params)
    model = ConvolutionalNeuralNetwork(Params)
    acccc = train(model,train_tasks,128)
    plt.plot(np.asarray(acccc))
    plt.ylabel('acc')
    plt.show()
