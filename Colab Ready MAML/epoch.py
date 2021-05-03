import model as MODEL
import utils
import train as TRAIN
import matplotlib.pyplot as plt
import torch


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
    plt.legend(loc = '.0006 LR rates both')
    plt.show()
    return Params, trainaccc, trianloss, dataacc, datatopk