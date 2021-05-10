import numpy as np
from numpy import save
import cv2 

#Note 44 images (2%) were wrong sized

filepath = 'UCMerced_LandUse/Images/'

image_savefile = "Extracted/UCMerced_images"
class_savefile = "Extracted/UCMerced_classes"
class_names_savefile = "Extracted/UCMerced_class_names"


scenes = ['agricultural',
            'airplane',
            'baseballdiamond',
            'beach',
            'buildings',
            'chaparral',
            'denseresidential',
            'forest',
            'freeway',
            'golfcourse',
            'harbor',
            'intersection',
            'mediumresidential',
            'mobilehomepark',
            'overpass',
            'parkinglot',
            'river',
            'runway',
            'sparseresidential',
            'storagetanks',
            'tenniscourt']


class_list = []
class_names = []
img_list = []
size = 256
channels = 3

for scene_num, scene_name in enumerate(scenes):
    print(scene_num, scene_name)
    class_names.append(scene_name)
    
    for idx in range(0,100):
        img_filename = filepath + scene_name + '/' + scene_name + str(idx).zfill(2) + '.tif'
        img = cv2.imread(img_filename)
        if not img.shape==(size,size,channels) :
            smin = min(img.shape[0], img.shape[1])
            img = img[:smin, :smin]
            img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
        img_list.append(img)
        class_list.append(scene_num)

img_array = np.array(img_list)
class_array = np.array(class_list)
class_names_array = np.array(class_names)

print('img_array:', img_array.shape)
print('class_array:', class_array.shape)
print('class_names_array:', class_names_array.shape)

save(image_savefile, img_array)
save(class_savefile, class_array)
save(class_names_savefile, class_names_array)

