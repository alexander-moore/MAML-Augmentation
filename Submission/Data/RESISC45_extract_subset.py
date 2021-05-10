import numpy as np
from numpy import save
import cv2 

filepath = 'NWPU-RESISC45'

image_savefile = "RESISC45_images_subset"
class_savefile = "RESISC45_classes_subset"
class_names_savefile = "RESISC_class_names_subset"


scenes = [  'airport',
            'basketball_court',
            'bridge',
            'church',
            'circular_farmland',
            'cloud',
            'commercial_area',
            'desert',
            'ground_track_field',
            'industrial_area',
            'island',
            'lake',
            'meadow',
            'mountain',
            'palace',
            'railway',
            'railway_station',
            'rectangular_farmland',
            'roundabout',
            'sea_ice',
            'ship',
            'snowberg',
            'stadium',
            'terrace',
            'thermal_power_station',
            'wetland']


class_list = []
class_names = []
img_list = []

for scene_num, scene_name in enumerate(scenes):
    print(scene_num, scene_name)
    class_names.append(scene_name)
    
    for idx in range(1,701):
        img_filename = filepath + scene_name + '/' + scene_name + '_' + str(idx).zfill(3) + '.jpg'
        img = cv2.imread(img_filename)
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


