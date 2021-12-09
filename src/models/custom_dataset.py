""" Data generator class. """

from ast import literal_eval
from tensorflow import keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def preprocess_image(img_filepath, scale):
    """ Given the path of an image, preprocesses image according to paper.
    
    Image preprocessing steps:
    - Get correct file path
    - Load in image
    - Resize image (not mentioned in paper but required/assumed)
    - Scale image between 0 and 1
    - Normalize each channel with respect to the ImageNet dataset
    """

    # Get correct relative filepath 
    img_filepath = os.path.join("../../data/raw/CrisisMMD_v2.0/", img_filepath)

    # Load in image and resize
    img = image.load_img(img_filepath, target_size=scale)

    # Convert image object to array
    img = image.img_to_array(img)

    # Expand image array 
    img = np.expand_dims(img, axis=0)

    # Convert image from RGB to BGR then zero-center each channel with respect to the ImageNet dataset
    # img = preprocess_input(img, mode="torch")[0]    # mode="torch" means image will be scaled then normalized
    img = preprocess_input(img)[0]    # mode="torch" means image will be scaled then normalized

    return img


class Image_Data_Generator(keras.utils.Sequence):

    def __init__(self, data_df, batch_size, scale=224) :
        self.data_df = data_df
        self.batch_size = batch_size
        self.scale = (scale, scale)

    def __len__(self) :
        return (np.ceil(len(self.data_df) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        # # shuffle at the start of each new epoch
        # if idx == 0:
        #     self.data_df = self.data_df.sample(frac=1)

        # Read in image data
        batch_image_X = [preprocess_image(img_filepath, self.scale) for img_filepath in self.data_df.iloc[idx * self.batch_size : (idx+1) * self.batch_size]["image"]]

        # Read in labels for current batch
        batch_y = list(self.data_df.iloc[idx * self.batch_size : (idx+1) * self.batch_size]["onehot_label"].apply(literal_eval))

        return np.asarray(batch_image_X), np.asarray(batch_y)


class Multimodal_Data_Generator(keras.utils.Sequence):

    def __init__(self, data_df, batch_size, scale=224) :
        self.data_df = data_df
        self.batch_size = batch_size
        self.scale = (scale, scale)

    def __len__(self) :
        return (np.ceil(len(self.data_df) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        # Read in image data
        batch_image_X = [preprocess_image(img_filepath, self.scale) for img_filepath in self.data_df.iloc[idx * self.batch_size : (idx+1) * self.batch_size]["image"]]

        # Read in text data
        batch_text_X = list(self.data_df.iloc[idx * self.batch_size : (idx+1) * self.batch_size]["padded_sequence"].apply(literal_eval))

        # Read in labels for current batch
        batch_y = list(self.data_df.iloc[idx * self.batch_size : (idx+1) * self.batch_size]["onehot_label"].apply(literal_eval))

        return [np.asarray(batch_image_X), np.asarray(batch_text_X)], np.asarray(batch_y)
