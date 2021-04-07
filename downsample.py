import numpy as np
import cv2
import matplotlib.pyplot as plt
#############################################################
indir = ''
infile = 'data/RESISC45_images.npy'

downsample_to = 96


outfile = f'data/RESISC45_images_{downsample_to}.npy'

show_samples = True
n_samples = 3
#############################################################

images = np.load(indir+infile)
downsized = np.empty((images.shape[0], downsample_to, downsample_to, images.shape[3]), dtype='uint8')

for r in range(images.shape[0]):
    downsized[r] = cv2.resize(images[r], dsize=(downsample_to, downsample_to), interpolation=cv2.INTER_CUBIC)

if show_samples:
    fig = plt.figure(figsize=(10,10))
    for r in range(n_samples):
        sub=fig.add_subplot(n_samples,2,2*r+1)
        sub.imshow(images[r])
        sub=fig.add_subplot(n_samples,2,2*r+2)
        sub.imshow(downsized[r])
    plt.tight_layout()
    plt.show()
    print(images.shape, downsized.shape, images[0].shape, downsized[0].shape, type(images), type(downsized), images.dtype, downsized.dtype)

np.save(indir+outfile, downsized)