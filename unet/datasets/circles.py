from typing import Tuple, List

import numpy as np
import tensorflow as tf
import glob
from PIL import Image
import numpy as np

channels = 1
classes = 2


def load_data(count:int, dataset_path:str, augmentation_rotation:int , splits:Tuple[float]=(0.7, 0.2, 0.1), **kwargs) -> List[tf.data.Dataset]:
    return [tf.data.Dataset.from_tensor_slices(_build_samples(int(split * count), dataset_path, augmentation_rotation , **kwargs))
            for split in splits]


def _build_samples(sample_count:int,dataset_path:str, augmentation_rotation:int , **kwargs) -> Tuple[np.array, np.array]:
    
    
    return _create_image_and_mask(dataset_path, augmentation_rotation)




def _create_image_and_mask(dataset_path, augmentation_rotation = 0, resize = (256, 256)):
    '''
    Description:
        Get your images and masks. This fucntion will also augment your data according to the selected
        parameter. If you put 0 their will be no augmentation if you assign 1 than every image will be rotated
        at (0, 360) angles if you selected 2 than every image will rotate frm (0, 360) with a gap of 2 degrees.
        if you dont want any rotation than assign 0. 
        
    Caution:
        If you used your method to generate masks than remember name of image and masks should be same
    
    Input:
        dataset_path  (Str): Path of your dataset where (mask and image) folder is located
        augmentation  (Int): Gap between the angles from (0, 360)
        
    Output:
        image (nd-array) : Image
        mask  (nd-array) : Label
    
    '''
    
    training_images, training_masks = [], []
    labels = None
    images = glob.glob(f"{dataset_path}/image/*.*g")
    
    for i in images:
        image_name = i.rsplit("/",1)[1]
        img        = Image.open(i).resize(resize).convert("L")
        mask       = Image.open(f"{dataset_path}/mask/{image_name}").convert("L").resize(resize)
        h, w       = mask.size
        
        np_image = np.expand_dims((np.asarray(img)/255.0).astype('float32'), axis = 2)
        training_images.append(np_image)
        np_mask = np.expand_dims((np.asarray(mask)/255.0).astype('float32'),axis =2)
        training_masks.append(np_mask)
        
        if augmentation_rotation != 0:
            for i in range(0, 360, augmentation_rotation):
#                 np_image = (np.asarray(img.rotate(i))/255.0).astype('float32')
                np_image = np.expand_dims((np.asarray(img.rotate(i))/255.0).astype('float32'), axis = 2)
                training_images.append(np_image)
                rot_mask   = mask.rotate(i)
                h, w       = rot_mask.size
                np_mask    = np.expand_dims((np.asarray(rot_mask)/255.0).astype('float32'),axis =2)
                training_masks.append(np_mask)
                
    return np.asarray(training_images), np.asarray(training_masks)
