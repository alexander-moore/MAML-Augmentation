# official train-test-split maker
import numpy as np
import random
from sklearn.model_selection import train_test_split

np.random.seed(1)
random.seed(1)

data = np.load('data/RESISC45_images_96.npy')
labels = np.load('data/RESISC45_classes.npy')

xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size = .3, stratify = labels)


np.save('data/RESISC45_images_train.npy', xtrain)
np.save('data/RESISC45_labels_train.npy', ytrain)
np.save('data/RESISC45_images_test.npy', xtest)
np.save('data/RESISC45_labels_test.npy', ytest)

print('Hopefully it worked! :) don\'t run it twice that might break it. i dont know!')
