import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report #, top_k_accuracy_score
import numpy as np
import sklearn
import os

def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds, axis=1)[:,-n:]
    ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(ts.shape[0]):
      if ts[i] in best_n[i,:]:
        successes += 1
    return float(successes)/ts.shape[0]

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

    parameters = {'n_estimators': [450, 500, 750],
              'criterion': ['entropy'], 
              'max_depth': [45], 
              'min_samples_split': [2]}

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

    print(train_data.shape)
    print(val_data.shape)

    scores = ['accuracy', 'average_precision', 'mutual_info_score']

    if True:
    ## SVM
        model = sklearn.svm.SVC()
        parameters = {'C': [0.1, 1.0, 2.0, 3.0, 4.0],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'coef0': [0.0, 0.1, 1],
              'degree': [1,2,3]}
        clf = GridSearchCV(model, parameters)
        print(train_data.shape, train_y.shape)
        clf.fit(train_data, train_y)
        print('RFC: ', clf.best_params_)
        best = clf.best_params_
        Tmodel = sklearn.svm.SVC(C = best['C'], degree = best['degree'], kernel = best['kernel'], coef0 = best['coef0'])
        Tmodel.fit(train_data, train_y)
        y_pred = Tmodel.predict(val_data)

        val_acc = sklearn.metrics.accuracy_score(val_y, y_pred)
        val_amis = sklearn.metrics.adjusted_mutual_info_score(val_y, y_pred)
        val_f1 = sklearn.metrics.f1_score(val_y, y_pred, average = 'weighted')

        #val_t5 = top_k_accuracy_score(val_y, y_pred)
        #y_pdf = Tmodel.decision_function(val_data)
        #val_t5 = top_k_accuracy_score(val_y, y_pdf)

        print(val_acc, val_amis, val_f1)



    if False:
    ## RandomForestClassifier
        model = sklearn.ensemble.RandomForestClassifier()
        clf = GridSearchCV(model, parameters)
        print(train_data.shape, train_y.shape)
        clf.fit(train_data, train_y)
        print('RFC: ', clf.best_params_)
        best = clf.best_params_
        Tmodel = sklearn.ensemble.RandomForestClassifier(n_estimators = best['n_estimators'],
                                                        max_depth = best['max_depth'],
                                                        criterion = best['criterion'],
                                                        min_samples_split = best['min_samples_split'])

        Tmodel.fit(train_data, train_y)
        y_pred = Tmodel.predict(val_data)

        val_acc = sklearn.metrics.accuracy_score(val_y, y_pred)
        val_amis = sklearn.metrics.adjusted_mutual_info_score(val_y, y_pred)
        val_f1 = sklearn.metrics.f1_score(val_y, y_pred, average = 'weighted')
        #y_pdf = Tmodel.decision_function(val_data)
        #val_t5 = sklearn.metrics.top_k_accuracy_score(val_y, y_pdf)
        #val_bsl = sklearn.metrics.brier_score_loss(val_y, y_pred)

        print(val_acc, val_amis, val_f1)#, val_t5)

    if False:
    ## LinearClassifier
        model = sklearn.linear_model.LogisticRegression(max_iter = 2000)
        parameters = {'fit_intercept':[0,1]}
        clf = GridSearchCV(model, parameters)
        print(train_data.shape, train_y.shape)
        clf.fit(train_data, train_y)
        print('RFC: ', clf.best_params_)
        best = clf.best_params_
        Tmodel = sklearn.linear_model.LogisticRegression(fit_intercept = best['fit_intercept'])

        Tmodel.fit(train_data, train_y)
        y_pred = Tmodel.predict(val_data)

        val_acc = sklearn.metrics.accuracy_score(val_y, y_pred)
        val_amis = sklearn.metrics.adjusted_mutual_info_score(val_y, y_pred)
        val_f1 = sklearn.metrics.f1_score(val_y, y_pred, average = 'weighted')
        #val_bsl = sklearn.metrics.brier_score_loss(val_y, y_pred)

        print(val_acc, val_amis, val_f1)


def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True,
                         sample_weight=None, labels=None):
    """Top-k Accuracy classification score.
    This metric computes the number of times where the correct label is among
    the top `k` labels predicted (ranked by predicted scores). Note that the
    multilabel case isn't covered here.
    Read more in the :ref:`User Guide <top_k_accuracy_score>`
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores. These can be either probability estimates or
        non-thresholded decision values (as returned by
        :term:`decision_function` on some classifiers). The binary case expects
        scores with shape (n_samples,) while the multiclass case expects scores
        with shape (n_samples, n_classes). In the nulticlass case, the order of
        the class scores must correspond to the order of ``labels``, if
        provided, or else to the numerical or lexicographical order of the
        labels in ``y_true``.
    k : int, default=2
        Number of most likely outcomes considered to find the correct label.
    normalize : bool, default=True
        If `True`, return the fraction of correctly classified samples.
        Otherwise, return the number of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.
    labels : array-like of shape (n_classes,), default=None
        Multiclass only. List of labels that index the classes in ``y_score``.
        If ``None``, the numerical or lexicographical order of the labels in
        ``y_true`` is used.
    Returns
    -------
    score : float
        The top-k accuracy score. The best performance is 1 with
        `normalize == True` and the number of samples with
        `normalize == False`.
    See also
    --------
    accuracy_score
    Notes
    -----
    In cases where two or more labels are assigned equal predicted scores,
    the labels with the highest indices will be chosen first. This might
    impact the result if the correct label falls after the threshold because
    of that.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import top_k_accuracy_score
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_score = np.array([[0.5, 0.2, 0.2],  # 0 is in top 2
    ...                     [0.3, 0.4, 0.2],  # 1 is in top 2
    ...                     [0.2, 0.4, 0.3],  # 2 is in top 2
    ...                     [0.7, 0.2, 0.1]]) # 2 isn't in top 2
    >>> top_k_accuracy_score(y_true, y_score, k=2)
    0.75
    >>> # Not normalizing gives the number of "correctly" classified samples
    >>> top_k_accuracy_score(y_true, y_score, k=2, normalize=False)
    3
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_true = column_or_1d(y_true)
    y_type = type_of_target(y_true)
    y_score = check_array(y_score, ensure_2d=False)
    y_score = column_or_1d(y_score) if y_type == 'binary' else y_score
    check_consistent_length(y_true, y_score, sample_weight)

    if y_type not in {'binary', 'multiclass'}:
        raise ValueError(
            f"y type must be 'binary' or 'multiclass', got '{y_type}' instead."
        )

    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2

    if labels is None:
        classes = _unique(y_true)
        n_classes = len(classes)

        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of classes in 'y_true' ({n_classes}) not equal "
                f"to the number of classes in 'y_score' ({y_score_n_classes})."
            )
    else:
        labels = column_or_1d(labels)
        classes = _unique(labels)
        n_labels = len(labels)
        n_classes = len(classes)

        if n_classes != n_labels:
            raise ValueError("Parameter 'labels' must be unique.")

        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")

        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of given labels ({n_classes}) not equal to the "
                f"number of classes in 'y_score' ({y_score_n_classes})."
            )

        if len(np.setdiff1d(y_true, classes)):
            raise ValueError(
                "'y_true' contains labels not in parameter 'labels'."
            )

    if k >= n_classes:
        warnings.warn(
            f"'k' ({k}) greater than or equal to 'n_classes' ({n_classes}) "
            "will result in a perfect score and is therefore meaningless.",
            UndefinedMetricWarning
        )

    y_true_encoded = _encode(y_true, uniques=classes)

    if y_type == 'binary':
        if k == 1:
            threshold = .5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
    elif y_type == 'multiclass':
        sorted_pred = np.argsort(y_score, axis=1, kind='mergesort')[:, ::-1]
        hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)

    if normalize:
        return np.average(hits, weights=sample_weight)
    elif sample_weight is None:
        return np.sum(hits)
    else:
        return np.dot(hits, sample_weight)