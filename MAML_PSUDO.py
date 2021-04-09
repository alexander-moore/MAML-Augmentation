
import torch
## need a model
model = make_model()
## up to choice
meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## sample from a batch of tasks check out pytorch-meta on github
#  https://github.com/tristandeleu/pytorch-meta
## it also has handy fucntions for handling names of inner params for each task
# so all is updated at the end and not during each tasks inner loop.
outer_loss = 0
for task in batch: #pull a task
     ## sample for the task, we need to have the struture to allow the following:
    train_inputs, train_targets = task["inner"] ## pull from the same task independent sets same augments different data
    test_inputs, test targets = task["outer"] ## pull from the same task independent sets same augments diffrent data


    train_logit = model(train_input)
     ## pick reconstruction loss
     inner_loss = reconstruction_loss(train_logit,train_target)
     model.zero_grads()
     ## we only take the grads of the paramater asccociated with the specific task of inner model, the auto encoder.
     # keep these grads by create_graph
     grads=torch.autograd.grad(inner_loss, model.meta_params(), create_graph=True)
     Params = OrderedDict()
     for (name,param), grad in zip(model.meta_named_pars(), grads()):
          ## update each parms of this task corresponding to the inner loop, we always update the autoencoder
          params[name] = params- step_size * grad
    ## get loss for the outer loop, the classification
     test_logit = model(test_input, test_target)
     ## accumulate all of the task specific losses into the outer loops. += not just =
     outer_loss += Cross_entropy(test_logit, test_target)
## with all the tasks sampled for the inner and outer loop we back prop the outer loop and thus the whole model
# (we take hessian of "grads" variable# from before)
outer_loss.backward()
meta_optimizer.step()
## check out this paper on similar topic. maybe we can use some of thier methods and impove upon
# META-LEARNING UPDATE RULES FOR UNSUPERVISED REPRESENTATION LEARNING
## https://arxiv.org/pdf/1804.00222.pdf

