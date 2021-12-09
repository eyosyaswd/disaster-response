from ast import literal_eval
from custom_dataset import Multimodal_Data_Generator
from performance_metrics import get_performance_metrics
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Ignore tf info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore tf warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Ignore tf warning messages
import pandas as pd
import pickle


if __name__ == "__main__":

    TASK = "humanitarian"

    print("\nLoading in testing data...")

    # Read in testing data
    test_filepath = f"../../data/interim/task_{TASK}_test_preprocessed_text.csv"
    test_df = pd.read_csv(test_filepath)
    
    # Read in label_encoder preduced from training data and get number of classes in dataset
    le = pickle.load(open(f"../../data/interim/{TASK}_label_encoder.pickle", "rb"))
    num_classes = len(le.classes_)

    # Extract data and labels from dataset
    test_data_gen = Multimodal_Data_Generator(test_df, batch_size=1)
    test_y = np.asarray(list(test_df["onehot_label"].apply(literal_eval)))
    
    print("\nLoading in trained model...")

    # Load in trained model (can't save model as hdf5)
    trained_model = load_model(f"../../models-improved/multimodal/{TASK}/{TASK}.hdf5")
    
    # # PC won't let me save model so just recreate the architecture then load in trained weights instead
    # vgg16_model = VGG16(weights="imagenet")
    # fc2 = vgg16_model.get_layer("fc2").output
    # output_layer = Dense(num_classes, activation="softmax")(fc2)
    # model = Model(inputs=vgg16_model.input, outputs=output_layer)
    # checkpoint_filepath = f"../../models-improved/image/{TASK}/{TASK}_checkpoint"
    # trained_model = model.load_weights(checkpoint_filepath)

    print(trained_model.summary()) 

    print("\nPredicting testing data...")
    
    # Predict testing data using trained model
    pred_y = trained_model.predict(test_data_gen, batch_size=1)

    print("\nGetting performance metrics...")

    # Get performance metrics
    get_performance_metrics(test_y, pred_y, test_df, TASK, "multimodal-improved")
    