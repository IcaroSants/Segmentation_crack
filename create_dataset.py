import pandas as pd 
import glob
import re
import os

def get_mask(images_paths, dir_mask = 'Masks'):
    mask = []
    blacklist = []
    for image_path in images_paths:
        mask_path = re.sub('Images',dir_mask, image_path)
        if os.path.isfile(mask_path) and os.path.isfile(image_path):
           mask.append(mask_path)
        else:
            blacklist.append(image_path)
        
    for image_path in blacklist:
        images_paths.remove(image_path)
    return mask

positive_images = glob.glob('Concrete/Positive/Images/*.png')
positive_images+= glob.glob('Concrete/Positive/Images/*.jpg')

negative_images = glob.glob('Concrete/Negative/Images/*.jpg')
negative_images +=glob.glob('Concrete/Negative/Images/*.png')

positives_masks_images = get_mask(positive_images)
negative_masks_images = get_mask(negative_images, dir_mask='Mask')


images = positive_images + negative_images
masks = positives_masks_images + negative_masks_images

df = pd.DataFrame({'Images':images, 'Mask':masks})
df.to_csv('dataset.csv')
