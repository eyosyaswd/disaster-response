from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf


def preprocess_image(img_filepath):
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
    img = image.load_img(img_filepath, target_size=(224, 224))

    # Convert image object to array
    img = image.img_to_array(img)

    # Expand image array 
    img = np.expand_dims(img, axis=0)

    # Convert image from RGB to BGR then zero-center each channel with respect to the ImageNet dataset
    img = preprocess_input(img, mode="torch")[0]    # mode="torch" means image will be scaled then normalized

    return img


def preprocess_data(df, data_type, le=None, ohe=None):
    """ Preprocess the dataset passed in as a dataframe. 
    
    Preprocessing steps:
    - Create integer and onehot labels
    - Shuffle images
    - Load in all images
    - Scale images between 0 and 1 
    - Normalize each channel with respect to the ImageNet dataset
    """

    # Create label encoding for dataset; OneHotEncoder() can't take in strings so 
    # first do integer encoding then convert integer encoding into one-hot encoding
    if data_type == "train": 
        # if training data, create encodings from scratch (fit and transform)
        le, ohe = LabelEncoder(), OneHotEncoder(sparse=False)
        df["int_label"] = le.fit_transform(df["label"])
        df["onehot_label"] = ohe.fit_transform(np.array(df["int_label"]).reshape(-1,1)).tolist()
    else:
        # if val or test data, use the encoders that were fit on training data (only transform)
        df["int_label"] = le.transform(df["label"])
        df["onehot_label"] = ohe.transform(np.array(df["int_label"]).reshape(-1,1)).tolist()
 
    # Shuffle data
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Processes images
    X = np.array([preprocess_image(img_filepath) for img_filepath in df["image"]])

    return df, le, ohe, X


if __name__ == "__main__":

    TASK = "informative"       # "humanitarian" or "informative"
    SEED = 2021                 # Seed to be used for reproducability

    print("\nLoading in data...")

    # Specify location of tsv files containing dataset
    train_filepath = f"../../data/raw/crisismmd_datasplit_agreed_label/task_{TASK}_text_img_agreed_lab_train.tsv"
    val_filepath = f"../../data/raw/crisismmd_datasplit_agreed_label/task_{TASK}_text_img_agreed_lab_dev.tsv"
    test_filepath = f"../../data/raw/crisismmd_datasplit_agreed_label/task_{TASK}_text_img_agreed_lab_test.tsv"

    # Load in tsv files containing data
    train_df = pd.read_csv(train_filepath, sep="\t")
    val_df = pd.read_csv(val_filepath, sep="\t")
    test_df = pd.read_csv(test_filepath, sep="\t")

    print("\nPreprocessing data...")

    # Preprocess data
    train_df, le, ohe, train_X = preprocess_data(train_df, data_type="train")
    val_df, _, _, val_X = preprocess_data(val_df, data_type="val", le=le, ohe=ohe)
    test_df, _, _, test_X = preprocess_data(test_df, data_type="test", le=le, ohe=ohe)

    # Save label_encoder (to be used during testing later)
    pickle.dump(le, open(f"../../data/interim/{TASK}_label_encoder.pickle", "wb"))

    # Output preprocessed data
    train_df.to_csv(f"../../data/interim/task_{TASK}_train_preprocessed_image.csv", index=False)
    val_df.to_csv(f"../../data/interim/task_{TASK}_val_preprocessed_image.csv", index=False)
    test_df.to_csv(f"../../data/interim/task_{TASK}_test_preprocessed_image.csv", index=False)
    np.save(f"../../data/interim/task_{TASK}_train_preprocessed_image.npy", train_X)
    np.save(f"../../data/interim/task_{TASK}_val_preprocessed_image.npy", val_X)
    np.save(f"../../data/interim/task_{TASK}_test_preprocessed_image.npy", test_X)
