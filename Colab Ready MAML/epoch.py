import model as MODEL
import utils
import train as TRAIN
import matplotlib.pyplot as plt
import torch
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


def epochthrough(train_tasks,Params,val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL.ConvolutionalNeuralNetwork(Params)
    model = model.to(device)
    loss_rate=0
    dataacc = []
    datatopk = []
    trianloss = []
    trainaccc = []
    epoch_tracker = [] 
    for i in range(Params['epoch']):
        trainacc , trainloss = TRAIN.train(model,train_tasks,Params,val_loader)
        loss_rate, losstopk = utils.getvalErr(model,val_loader)
        print("epoch:",i,"val accuracy: ",loss_rate, ' topk: ', losstopk)
        dataacc.append(loss_rate)
        datatopk.append(losstopk)
        trianloss.append(trainloss)
        trainaccc.append(trainacc)
        epoch_tracker.append(i)

    plt.plot(epoch_tracker, trainaccc, label = 'train acc')
    plt.plot(epoch_tracker, dataacc, label = 'val acc')
    plt.plot(epoch_tracker, datatopk, label = 'val top3')
    string = str(Params['innerStep']) + "LR rate "
    if Params['Alex'] == False:
        string+=" inner loop, "+ str(Params['MetaLR']) + ' LR outer loop, '
        if Params['Quincy'] == False:
            string+= str(Params['outerVSinner'])+' to 1 ratio, '+ str(Params['nways']) +" ways, " + str(Params['kshots']) +" shots,"+str(Params['number_of_tasks']) + " task per outer, " + "first order "
            if Params['Order']==True:
                string+="True"
            else:
                string+= "False"
    plt.legend()
    plt.title(string)
    plt.show()
    torch.save(model, '/content/drive/MyDrive/Project/genMAML.pt')
    return Params, trainaccc, trianloss, dataacc, datatopk


def epochMercer(Params,ptfilepath='/content/drive/MyDrive/Project/genMAML.pt'):
    Params['mercer'] = True
    Params['augs'] = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(ptfilepath)
    model = model.to(device)
    Params['Alex']=True ## set this to see how it transfers on CNN
    Params['innerStep'] = Params['transferLR']
    model.classifier = MetaLinear(model.hidden_size**10, model.out_features)
    if Params['freeze']==True:
        for param in model.features.parameters(): ## freeze conv
            param.requires_grad = False
    train_tasks,val_loader,test_loader,Params=utils.colabgetdata(Params)
    loss_rate=0
    dataacc = []
    datatopk = []
    trianloss = []
    trainaccc = []
    epoch_tracker = [] 
    for i in range(50):
        trainacc , trainloss = TRAIN.train(model,train_tasks,Params,val_loader)
        loss_rate, losstopk = utils.getvalErr(model,val_loader)
        print("epoch:",i,"val accuracy: ",loss_rate, ' topk: ', losstopk)
        dataacc.append(loss_rate)
        datatopk.append(losstopk)
        trianloss.append(trainloss)
        trainaccc.append(trainacc)
        epoch_tracker.append(i)

    plt.plot(epoch_tracker, trainaccc, label = 'train acc')
    plt.plot(epoch_tracker, dataacc, label = 'val acc')
    plt.plot(epoch_tracker, datatopk, label = 'val top3')
    string = "MERCER "+str(Params['innerStep']) + "LR rate "
    if Params['Alex'] == False:
        string+=" inner loop, "+ str(Params['MetaLR']) + ' LR outer loop, '
        if Params['Quincy'] == False:
            string+= str(Params['outerVSinner'])+' to 1 ratio, '+ str(Params['nways']) +" ways, " + str(Params['kshots']) +" shots,"+str(Params['number_of_tasks']) + " task per outer, " + "first order "
            if Params['Order']==True:
                string+="True"
            else:
                string+= "False"
    plt.title(string)
    plt.legend()
    plt.show()
    return Params, trainaccc, trianloss, dataacc, datatopk