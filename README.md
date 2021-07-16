# Model-Agnostic Meta Learning for Remote Sensing

Remote sensing is a natural image classification task in which the image content may be at drastically different scales.

Remote sensing scene classification for this project classifies images taken from airborne or spaceborne sensors, creating challenges for previous work including:
- Poor generalization across datasets
- Data variance between training and testing sets
- Low-data paradigms reliant on fine-tuning from pretrained natural image sets

## Why Meta Learning?

Meta Learning is well-developed in supervised natural image tasks, though we are not aware of present research which relies on augmentations to create tasks for a natural image classification task. **We hypothesize that training a meta learner using compositions of image augmentations may lead to a more robust, generalizable model**. Model-agnostic meta learning (MAML) offers improved model performance in few-shot training scenarios and may be able to quickly optimize parameters to datasets with new classes with little fine-tuning.

## Novelty
We induce tasks for the meta learner to simultaneously opimize by compositing augmentations. This composition is given by 

![augment image](imgs/aug_table.png)

## Data Set
* NWPU - RESISC45
    - 31,500 Google Earth (256x256x3) images over 45 classes

* UC Merced Land Use
     - 2,100 USGS National Map Urban Imagery (256x256x3) images over 21 classes

## Challenges
- Image variance in scale and resolution
- Inter- and intra-class variance
- Multiple objects within scenes
- High number of classes
- Resizing dataset reduces discriminability
- Class overlap in style and content

