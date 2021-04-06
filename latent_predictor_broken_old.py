import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def fit_optim_predictor(model, xtrain, ytrain, xtest, ytest):
	# Pass an AE, train val test image sets, and corresp labels

	# Labels should be the default data format (integers, not one-hot)

	# Returns:
	# Given an autoencoder model, optimize a classifier in the latent space over a gridsearch validation
	# Return this classifier and it's scores

	xtrain = model.encode(xtrain).squeeze()
	xtest = model.encode(xtest).squeeze()

	tuned_parameters = [{'n_estimators': [50, 100, 150], 'min_samples_split': [2, 8, 32], 'max_depth': [1,2,4]}]

	scores = ['f1_score']

	for score in scores:
	    print("# Tuning hyper-parameters for %s" % score)
	    print()

	    clf = GridSearchCV(
	        RandomForestClassifier(), tuned_parameters, scoring='%s_macro' % score
	    )
	    clf.fit(xtrain, ytrain)

	    print("Best parameters set found on development set:")
	    print()
	    print(clf.best_params_)
	    print()
	    print("Grid scores on development set:")
	    print()
	    means = clf.cv_results_['mean_test_score']
	    stds = clf.cv_results_['std_test_score']
	    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	        print("%0.3f (+/-%0.03f) for %r"
	              % (mean, std * 2, params))
	    print()

	    print("Detailed classification report:")
	    print()
	    print("The model is trained on the full development set.")
	    print("The scores are computed on the full evaluation set.")
	    print()
	    y_true, y_pred = ytest, clf.predict(xtest)
	    final_report = classification_report(y_true, y_pred)
	    print(final_report)
	    print()

	return clf, final_report