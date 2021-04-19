import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

import sklearn
import os

def fit_optim_predictor(AE, train_data, train_labels, test_data, test_labels):
    # Pass an AE, train val test image sets, and corresp labels

    # Labels should be the default data format (integers, not one-hot)

    # Returns:
    # Given an autoencoder model, optimize a classifier in the latent space over a gridsearch validation
    # Return this classifier and it's scores
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xtrain, xval, ytrain, yval = train_test_split(train_data, train_labels, test_size = 0.25)
    bs = 64
    xtrain = torch.tensor(xtrain).permute(0,3,1,2)
    print(torch.min(xtrain), torch.max(xtrain))

    if os.path.isfile('train_embed.pt'):
    #if False:
    	train_embedded = torch.load('train_embed.pt')
    	train_y = torch.load('train_y.pt')

    	val_embedded = torch.load('val_embed.pt')
    	val_y = torch.load('val_y.pt')

    	test_embedded = torch.load('test_embed.pt')
    	test_y = torch.load('test_y.pt')


    else:
	    trainset = []
	    for i in range(xtrain.shape[0]):
	        trainset.append((xtrain[i], ytrain[i]))
	        
	    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
	                                              shuffle=False) #BUG: must keep shuffle false - or else it screws up labels, apparently

	    ## Validation Data
	    valset = []
	    xval = torch.tensor(xval).permute(0,3,1,2)

	    print(torch.min(xval), torch.max(xval))
	    for i in range(xval.shape[0]):
	        valset.append((xval[i], yval[i]))

	    val_loader = torch.utils.data.DataLoader(valset, batch_size=1, drop_last = True,
	                                              shuffle=False) #BUG: must keep shuffle false - or else it screws up labels, apparently

	    

	    test_data = torch.tensor(test_data)
	    test_labels = torch.tensor(test_labels)
	    ## Testing Data
	    testset = []
	    print(test_data.shape)
	    xtest = torch.tensor(test_data).permute(0,3,1,2)
	    print(xtest.shape)

	    print(torch.min(test_data), torch.max(test_data))
	    for i in range(test_data.shape[0]):
	        testset.append((test_data[i], test_labels[i]))

	    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, drop_last = True,
	                                              shuffle=False) #BUG: must keep shuffle false - or else it screws up labels, apparently

	    # Embed Train set points:
	    embedded_points = []
	    labels = []

	    AE = AE.eval().to(device)

	    with torch.no_grad():
	        for x,y in train_loader:
	            x = 2*((x / 255) - 0.5)
	            x = AE.encode(x.to(device)).squeeze()
	            if x.shape[0] == 54:
	                break
	            embedded_points.append(x)
	            labels.append(y)

	    embedded_points = embedded_points[:-1] #skip last bad shape
	    labels = labels[:-1]
	    train_embedded = torch.cat(embedded_points, dim=0)
	    train_y = torch.cat(labels, dim = 0)
	    print('train set shape:')
	    print(train_embedded.shape)
	    print(train_y.shape)

	    torch.save(train_embedded, 'train_embed.pt')
	    torch.save(train_y, 'train_y.pt')

	    # Embed Validation set points:
	    embedded_points = []
	    labels = []

	    with torch.no_grad():
	        for x,y in val_loader:
	            x = 2*((x / 255) - 0.5)
	            
	            x = AE.encode(x.to(device))
	            embedded_points.append(x)
	            labels.append(y)
	            
	    val_embedded = torch.stack(embedded_points).squeeze()
	    val_y = torch.stack(labels).squeeze()
	    print('validation set shape: ', val_embedded.shape)
	    print(val_y.shape)

	    torch.save(val_embedded, 'val_embed.pt')
	    torch.save(val_y, 'val_y.pt')

	    # Embed test set points:
	    embedded_points = []
	    labels = []

	    print(next(iter(test_loader))[0].shape)

	    with torch.no_grad():
	        for x,y in test_loader:
	            x = x.permute(0,3,1,2)
	            
	            x = 2*((x / 255) - 0.5)
	            
	            x = AE.encode(x.to(device))
	            embedded_points.append(x)
	            labels.append(y)
	            
	    test_embedded = torch.stack(embedded_points).squeeze()
	    test_y = torch.stack(labels).squeeze()
	    print('test set shape:')
	    print(test_embedded.shape)
	    print(test_y.shape)

	    torch.save(test_embedded, 'test_embed.pt')
	    torch.save(test_y, 'test_y.pt')

    tuned_parameters = [{'n_estimators': [100, 150, 200], 'min_samples_split': [1,2,4,8], 'max_depth': [2,3,4,5]}]

    #scores = [sorted(sklearn.metrics.SCORERS.keys())]
    #"['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 
    #'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'max_error', 'mutual_info_score', 
    #'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 
    #'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 
    #'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'v_measure_score']_macro"

    n_samps = 2500

    train_data = train_embedded[0:n_samps].cpu()
    train_y = train_y[0:n_samps].cpu()

    val_data = val_embedded[0:n_samps].cpu()
    val_y = val_y[0:n_samps].cpu()

    scores = ['accuracy', 'average_precision', 'mutual_info_score']

    if True:
    ## RandomForestClassifier
        model = sklearn.ensemble.RandomForestClassifier()
        parameters = tuned_parameters
        clf = GridSearchCV(model, parameters)
        print(train_data.shape, train_y.shape)
        clf.fit(train_data, train_y)
        print('RFC: ', clf.best_params_)
        best = clf.best_params_
        Tmodel = sklearn.ensemble.RandomForestClassifier(n_estimators = best['n_estimators'],
                                                        max_depth = best['max_depth'],
                                                        min_samples_split = best['min_samples_split'])

        Tmodel.fit(train_data, train_y)
        y_pred = Tmodel.predict(val_data)

        val_acc = sklearn.metrics.accuracy_score(val_y, y_pred)
        val_amis = sklearn.metrics.adjusted_mutual_info_score(val_y, y_pred)
        val_f1 = sklearn.metrics.f1_score(val_y, y_pred, average = 'weighted')
        val_bsl = sklearn.metrics.brier_score_loss(val_y, y_pred)

        print(val_acc, val_amis, val_f1, val_bsl)

    if False:
    ## LinearClassifier
        model = sklearn.linear_model.RidgeClassifierCV()
        parameters = {'alphas':[0, 0.001, 0.01, 0.1, 0.25], 
                      'fit_intercept':[0,1]}
        clf = GridSearchCV(model, parameters)
        print(train_data.shape, train_y.shape)
        clf.fit(train_data, train_y)
        print('RFC: ', clf.best_params_)
        best = clf.best_params_
        Tmodel = sklearn.ensemble.RandomForestClassifier(alphas = best['alphas'],
                                                        fit_intercept = best['fit_intercept'])

        Tmodel.fit(train_data, train_y)
        y_pred = Tmodel.predict(val_data)

        val_acc = sklearn.metrics.accuracy_score(val_y, y_pred)
        val_amis = sklearn.metrics.adjusted_mutual_info_score(val_y, y_pred)
        val_f1 = sklearn.metrics.f1_score(val_y, y_pred, average = 'weighted')
        val_bsl = sklearn.metrics.brier_score_loss(val_y, y_pred)

        print(val_acc, val_amis, val_f1, val_bsl)