import torch
import utils
import augmentation_functions
from torchmeta.utils.gradient_based import gradient_update_parameters
import torch.nn.functional as F


def train(model, train_tasks, Params, val_loader):  ## go through data set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=Params['MetaLR'])
    outer_loss = torch.tensor(0., device=device)
    model.zero_grad()
    j = 0  # keeping track of tasks seen
    trainacc = []
    trainloss = []
    for task in train_tasks:
        i = torch.randint(high=127, size=(1, 1)).item()
        ## call augments on the fly
        x, y = task
        x = x.float().to(device)
        y = y.long().to(device)
        if Params['aug'] == False:
            x_inner, x_outer, y_inner, y_outer = utils.tasksplit(x, y, Params)  ## split task for inner/ ouyter with no augs
        else:
            x_inner, x_outer, y_inner, y_outer = utils.tasksplit(augmentation_functions.applyAugs(x, i), y,
                                                           Params)  ## split task for inner/ ouyter with same augs
        train_logit = model(x_inner)
        inner_loss = F.cross_entropy(train_logit, y_inner)
        model.zero_grad()
        params = gradient_update_parameters(model,
                                            inner_loss,
                                            # params=None, ## note we can do a for loop and take multiple inner loop steps, set to none since
                                            step_size=Params["innerStep"],
                                            first_order=Params["Order"])
        test_logit = model(x_outer,
                           params=params)  ## take the loss fucntions using the params of this task specific inner loop
        current_outer_loss = F.cross_entropy(test_logit, y_outer)
        outer_loss += current_outer_loss
        current_outer_loss.div_(Params['nways'] * Params['kshots'])
        acc = utils.get_accuracy(test_logit, y_outer)
        trainacc.append(acc)
        trainloss.append(current_outer_loss)

        j += 1
        if j % Params['number_of_tasks'] == 0:  ## we hit number of tasks if this =0
            outer_loss.backward()
            meta_optimizer.step()
            outer_loss = torch.tensor(0., device=device)
    return torch.mean(torch.FloatTensor(trainacc)).data.item(), torch.sum(torch.FloatTensor(trainloss)).data.item()


