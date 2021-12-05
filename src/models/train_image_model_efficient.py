from ast import literal_eval

from tensorflow.keras.applications import efficientnet
from custom_dataset import Image_Data_Generator
from tensorflow.keras.applications.efficientnet import EfficientNetB7, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Ignore tf info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore tf warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Ignore tf warning messages
import pandas as pd
import pickle
import sys
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if __name__ == "__main__":

    TASK = "informative"       # "humanitarian" or "informative"
    SEED = 2021                # Seed to be used for reproducability

    print("\nLoading in dataset...")

    # Specify location of csv files containing preprocessed data
    train_filepath = f"../../data/interim/task_{TASK}_train_preprocessed_image.csv"
    val_filepath = f"../../data/interim/task_{TASK}_val_preprocessed_image.csv"

    # Load in csv files containing preprocessed data
    train_df = pd.read_csv(train_filepath)
    val_df = pd.read_csv(val_filepath)

    # Load in image data generators
    train_data_gen = Image_Data_Generator(train_df, batch_size=4, scale=600)
    val_data_gen = Image_Data_Generator(val_df, batch_size=4, scale=600)

    # Get the number of classes
    num_classes = len(train_df["int_label"].unique())

    print("\nCreating EfficientNetB7 model...")

    # Create EfficientNetB7 model
    # model = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(224, 224, 3), classes=num_classes)
    efficient_model = EfficientNetB7(weights="imagenet")
    # print(efficient_model.summary())
    # sys.exit()

    # Change last layer from 1000 outputs to num_classes
    predictions = efficient_model.get_layer("predictions").output
    output_layer = Dense(num_classes, activation="softmax")(predictions)
    model = Model(inputs=efficient_model.input, outputs=output_layer)

    print(model.summary())

    # Initialize Adam optimizer
    adam = Adam(learning_rate=1e-4)
    
    # Config model with losses and metrics
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    # Initialize learning rate reducer
    lr_reducer = ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=1, verbose=1, mode="max")

    # Set early-stopping criterion based on the accuracy on the development set with the patience of 10
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, mode="max")

    # Initialize TensorBoard to visualize learning
    tensorboard = TensorBoard(log_dir=f"../../models-improved/image/{TASK}", write_images=True)

    # Create model checkpoints
    checkpoint_filepath = f"../../models-improved/image/{TASK}/{TASK}_checkpoint"
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="max")

    # Train and validate model
    history = model.fit(x=train_data_gen, epochs=10, validation_data=val_data_gen, callbacks=[lr_reducer, early_stopping, tensorboard, checkpoint])

    # Load model with best weights
    model.load_weights(checkpoint_filepath)

    # Save trained model with best weights
    model.save(f"../../models-improved/image/{TASK}/{TASK}.hdf5")
