import numpy as np
from numpy import save
import cv2 

filepath = 'NWPU-RESISC45/'

image_savefile = "RESISC45_images"
class_savefile = "RESISC45_classes"
class_names_savefile = "RESISC45_class_names"


scenes = ['airplane',
            'airport',
            'baseball_diamond',
            'basketball_court',
            'beach',
            'bridge',
            'chaparral',
            'church',
            'circular_farmland',
            'cloud',
            'commercial_area',
            'dense_residential',
            'desert',
            'forest',
            'freeway',
            'golf_course',
            'ground_track_field',
            'harbor',
            'industrial_area',
            'intersection',
            'island',
            'lake',
            'meadow',
            'medium_residential',
            'mobile_home_park',
            'mountain',
            'overpass',
            'palace',
            'parking_lot',
            'railway',
            'railway_station',
            'rectangular_farmland',
            'river',
            'roundabout',
            'runway',
            'sea_ice',
            'ship',
            'snowberg',
            'sparse_residential',
            'stadium',
            'storage_tank',
            'tennis_court',
            'terrace',
            'thermal_power_station',
            'wetland']


class_list = []
class_names = []
img_list = []

for scene_num, scene_name in enumerate(scenes):
    print(scene_num, scene_name)
    class_names.append(scene_name)
    
    for idx in range(1,700):
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


