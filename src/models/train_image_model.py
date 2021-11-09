from ast import literal_eval
from tensorflow.keras.applications.vgg16 import VGG16
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

    # Load in images and extract labels from dataset
    train_X = np.load(f"../../data/interim/task_{TASK}_train_preprocessed_image.npy", allow_pickle=True)
    train_y = np.asarray(list(train_df["onehot_label"].apply(literal_eval)))
    val_X = np.load(f"../../data/interim/task_{TASK}_val_preprocessed_image.npy", allow_pickle=True)
    val_y = np.asarray(list(val_df["onehot_label"].apply(literal_eval)))

    # Get the number of classes
    # print(train_df["label"])
    num_classes = len(train_df["int_label"].unique())

    print("\nCreating VGG16 model...")

    # Create VGG16 model
    vgg16_model = VGG16(weights="imagenet")

    # Change last layer from 1000 outputs to num_classes
    fc2 = vgg16_model.get_layer("fc2").output
    output_layer = Dense(num_classes, activation="softmax")(fc2)
    model = Model(inputs=vgg16_model.input, outputs=output_layer)

    print(model.summary())

    # Initialize Adam optimizer
    adam = Adam(learning_rate=1e-6)
    
    # Config model with losses and metrics
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    # Initialize learning rate reducer
    lr_reducer = ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=10, verbose=1, mode="max")

    # Set early-stopping criterion based on the accuracy on the development set with the patience of 10
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, mode="max")

    # Initialize TensorBoard to visualize learning
    tensorboard = TensorBoard(log_dir=f"../../models/image/{TASK}", write_images=True)

    # Create model checkpoints
    checkpoint_filepath = f"../../models/image/{TASK}/{TASK}_checkpoint"
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="max")

    # Train and validate model
    history = model.fit(x=train_X, y=train_y, batch_size=1, epochs=10, validation_data=(val_X, val_y), validation_batch_size=1, callbacks=[lr_reducer, early_stopping, tensorboard, checkpoint])

    # Load model with best weights
    model.load_weights(checkpoint_filepath)

    # Save trained model with best weights
    model.save(f"../../models/image/{TASK}/{TASK}.hdf5")
