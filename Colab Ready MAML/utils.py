import torch
import numpy as np
import learn2learn as l2l
import math
import torch.utils
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def taskStructure(dataset,Params,train_labels):
    trainDATA = OurDataset(dataset)
    labels= {}
    for i in range(45):
        labels[i] = [j for j, x in enumerate(train_labels) if x == i]
    metaDataset = l2l.data.MetaDataset(trainDATA,labels_to_indices=labels)
    if Params['Quincy'] == False and Params['Alex']==False:
        transforms = [
        ## need kshots*2 for inner and outer loop split accross each task
            l2l.data.transforms.FusedNWaysKShots(metaDataset,n=Params['nways'],k=Params['kshots']*(Params['outerVSinner']+1),replacement=False),
            l2l.data.transforms.LoadData(metaDataset),
            l2l.data.transforms.ConsecutiveLabels(metaDataset)
        ]
    elif Params['Quincy']==True:
        print("doing Quincy's model")
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=int(Params['bs']*2), shuffle=True) #BUG: must keep shuffle false - or else it screws up labels, apparently
        return train_loader
    elif Params['Alex']==True:
        print("doing Alex's model")
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=int(Params['bs']), shuffle=True) #BUG: must keep shuffle false - or else it screws up labels, apparently
        return train_loader


    taskset = l2l.data.TaskDataset(metaDataset,
                                   transforms,
                                   num_tasks=math.floor(Params['trainsz']/((Params['nways']*Params['kshots']*(Params['outerVSinner']+1))))
                                   ) ## generate max number of tasks
    return taskset


class OurDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset: the training set
        """
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        return sample
class OurDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset: the training set
        """
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        return sample

class OurDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset: the training set
        """
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        return sample

def linkDataset(x,y):
    set = []
    for i in range(len(y)):
        set.append((x[i], y[i]))
    return set
def getvalErr(model,val_loader):
    val_accs = []
    val_topks = []
    with torch.no_grad():
        accs, actk = [], []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for x, y in val_loader:
            x, y = x.to(device).float(), y.to(device).long()
            yhat = model(x)
            yhat_max = torch.max(yhat, dim=1)[1]
            correct = torch.sum(yhat_max == y)
            size = x.shape[0]
            acc_topk = accuracy_topk(yhat, y)
            actk.append(acc_topk.data.item())
            valcc = (correct / size).data.item()
            accs.append(valcc)
    return torch.mean(torch.FloatTensor(accs)).data.item(), torch.mean(torch.FloatTensor(actk)).data.item()

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
        res.append(correct_k.mul_(1 / batch_size))
    return res[0]
def tasksplit(x,y,Params):
    ## split inner and outer loop by incidies
    skip = Params['outerVSinner']+1
    idx = idx = torch.arange(start =1, end=x.shape[0]+1).bool()
    idx[::skip] = False
    x_inner = x[idx]
    y_inner = y[idx]
    x_outer = x[~idx]
    y_outer = y[~idx]
    return x_inner, x_outer, y_inner,y_outer
def colabgetdata(Params, load = True):
    np.random.seed(1)
    random.seed(1)
    # file = 'data/RESISC45_images_'+str(Params['datasize'])+'.npy'
    if Params['mercer']==True:
        print('UCmercer')
        path = Params['path']
        file = path+ 'Project/data/UCMerced_images96.npy'
        if load:
            data = (np.load(file)/127.5)-1
            print(data.shape)
            print(data.shape)
            labels = np.load(path+'Project/data/UCMerced_classes.npy')
            xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=Params['traintestSplit'], stratify=labels)

            np.save(path + 'Project/data/UCMerced_images_train.npy', xtrain)
            np.save(path + 'Project/data/UCMerced_labels_train.npy', ytrain)
            np.save(path+ 'Project/data/UCMerced_images_test.npy', xtest)
            np.save(path+ 'Project/data/UCMerced_labels_test.npy', ytest)
        else:
            train_data = np.load(path+'Project/data/UCMerced_images_train.npy')
            train_labels = np.load(path+'/Project/data/UCMerced_labels_train.npy')
            classes = np.load(path+ 'Project/data/UCMerced_class_names.npy')
    else:
        path = Params['path']
        if load:
            file = path + 'Project/data/RESISC45_images_96.npy'
            data = (np.load(file)/127.5)-1
            labels = np.load(path+ 'Project/data/RESISC45_classes.npy')
            test_size = 0.25
            xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=Params['traintestSplit'], stratify=labels)

            np.save(path+'Project/data/RESISC45_images_train.npy', xtrain)
            np.save(path+'Project/data/RESISC45_labels_train.npy', ytrain)
            np.save(path+'Project/data/RESISC45_images_test.npy', xtest)
            np.save(path+'Project/data/RESISC45_labels_test.npy', ytest)
        else:
            train_data = np.load(path+ 'Project/data/RESISC45_images_train.npy')
            train_labels = np.load(path+'Project/data/RESISC45_labels_train.npy')
            classes = np.load(path+ 'Project/data/RESISC45_class_names.npy')
    img_size = train_data.shape[2]  # can use this to mofidy data size to fit this model (which only takes 256 images)
    c_dim = classes.shape[0]
    xtrain, xval, ytrain, yval = train_test_split(train_data, train_labels, test_size=Params['trainvalSplit'])
    Params["trainsz"] = len(xtrain)
    bs=32
    Params['bs'] = bs

    xtrain = torch.tensor(xtrain).permute(0, 3, 1, 2)
    trainset = linkDataset(xtrain,ytrain)
    train_tasks = taskStructure(trainset,Params,ytrain)
    ## Validation Data
    valset = []
    xval = torch.tensor(xval).permute(0, 3, 1, 2)
    valset = linkDataset(xval,yval)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=32, drop_last=True,
                                             shuffle=False)  # BUG: must keep shuffle false - or else it screws up labels, apparently
    if Params['mercer']==True:
        test_data = np.load(path+'Project/data/UCMerced_images_test.npy')
        test_labels = np.load(path+'Project/data/UCMerced_labels_test.npy')
    else:
        test_data = np.load(path+'Project/data/RESISC45_images_test.npy')
        test_labels = np.load(path+'Project/data/RESISC45_labels_test.npy')
    test_data = torch.tensor(test_data)
    test_labels = torch.tensor(test_labels)
    ## Testing Data
    xtest = torch.tensor(test_data).permute(0, 3, 1, 2)
    testset = linkDataset(xtest,test_labels)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, drop_last=True,
                                              shuffle=False) # BUG: must keep shuffle false - or else it screws up labels, apparently
    Params['Img_size'] = img_size
    Params["c_dim"] = c_dim


    return train_tasks,val_loader,test_loader, Params
def make_example(train_loader):
    print('Shape of a batch of images: ', next(iter(train_loader))[0].shape)
    print('Shape of a batch of labels: ', next(iter(train_loader))[1].shape)

    first_samp = next(iter(train_loader))[0][0] #get first sample in first batch
    plt.imshow(first_samp.permute(1,2,0)/255) #show it
def make_train_plot(MAML_loss_tracker, CNN_loss_tracker, MAML_val_accs, CNN_val_accs, MAML_val_topks, CNN_val_topks, epoch_tracker,Params):
    #plt.plot(epoch_tracker, MAML_loss_tracker, label = 'MAML train loss')
    #plt.plot(epoch_tracker, CNN_loss_tracker, label = 'CNN train loss')
    plt.plot(epoch_tracker, MAML_val_accs, label = 'MAML val acc')
    plt.plot(epoch_tracker, CNN_val_accs, label = 'CNN val acc')
    plt.plot(epoch_tracker, MAML_val_topks, label = 'MAML val top3')
    plt.plot(epoch_tracker, CNN_val_topks, label = 'CNN_val top3')
    plt.legend(loc = 'best')
    string = str(Params['innerStep']) + "LR rate "
    if Params['Alex'] == False:
        string+=" inner loop, "+ str(Params['MetaLR']) + ' LR outer loop, '
        if Params['Quincy'] == False:
            string+= str(Params['outerVSinner'])+' to 1 ratio, '+ str(Params['nways']) +" ways, " + str(Params['kshots']) +" shots,"+str(Params['number_of_tasks']) + " task per outer, " + "first order "
            if Params['Order']==True:
                string+="True"
            else:
                string+= "False"
    plt.title(string)
    plt.show()

def make_two_plots(epoch_tracker, MAML_loss_tracker, CNN_loss_tracker, MAML_val_accs, CNN_val_accs,Params):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(epoch_tracker, MAML_loss_tracker, label = 'MAML train loss')
    ax1.plot(epoch_tracker, CNN_loss_tracker, label = 'CNN train loss')
    ax1.legend(loc = 'best')
    ax2.plot(epoch_tracker, MAML_val_accs, label = 'MAML val acc')
    ax2.plot(epoch_tracker, CNN_val_accs, label = 'CNN val acc')
    ax2.legend(loc = 'best')
    string = str(Params['innerStep']) + "LR rate "
    if Params['Alex'] == False:
        string+=" inner loop, "+ str(Params['MetaLR']) + ' LR outer loop, '
        if Params['Quincy'] == False:
            string+= str(Params['outerVSinner'])+' to 1 ratio, '+ str(Params['nways']) +" ways, " + str(Params['kshots']) +" shots,"+str(Params['number_of_tasks']) + " task per outer, " + "first order "
            if Params['Order']==True:
                string+="True"
            else:
                string+= "False"
    plt.title(string)
    plt.tight_layout()
    plt.show()

def basic_train_plot(epoch_tracker, MAML_loss_tracker, CNN_loss_tracker,Params):
    plt.plot(epoch_tracker, MAML_loss_tracker, label = 'MAML train loss')
    plt.plot(epoch_tracker, CNN_loss_tracker, label = 'CNN train loss')
    plt.legend(loc = 'best')
    string = str(Params['innerStep']) + "LR rate "
    if Params['Alex'] == False:
        string+=" inner loop, "+ str(Params['MetaLR']) + ' LR outer loop, '
        if Params['Quincy'] == False:
            string+= str(Params['outerVSinner'])+' to 1 ratio, '+ str(Params['nways']) +" ways, " + str(Params['kshots']) +" shots,"+str(Params['number_of_tasks']) + " task per outer, " + "first order "
            if Params['Order']==True:
                string+="True"
            else:
                string+= "False"
    plt.title(string)
    plt.show()
