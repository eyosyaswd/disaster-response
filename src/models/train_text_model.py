from ast import literal_eval
from sentence_cnn import SentenceCNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Ignore tf info messages
import pandas as pd
import pickle


if __name__ == "__main__":

    TASK = "informative"       # "humanitarian" or "informative"
    SEED = 2021                # Seed to be used for reproducability

    print("\nLoading in dataset, word_index, and embedding_matrix...")

    # Specify location of csv files containing preprocessed data
    train_filepath = f"../../data/interim/task_{TASK}_train_preprocessed_text.csv"
    val_filepath = f"../../data/interim/task_{TASK}_val_preprocessed_text.csv"
    test_filepath = f"../../data/interim/task_{TASK}_test_preprocessed_text.csv"

    # Load in csv files containing preprocessed data
    train_df = pd.read_csv(train_filepath)
    val_df = pd.read_csv(val_filepath)
    test_df = pd.read_csv(test_filepath)

    # Extract data and labels from dataset
    train_X = list(train_df["padded_sequence"].apply(literal_eval))
    train_y = list(train_df["onehot_label"].apply(literal_eval))
    val_X = list(val_df["padded_sequence"].apply(literal_eval))
    val_y = list(val_df["onehot_label"].apply(literal_eval))

    # Get the number of classes
    # print(train_df["label"])
    num_classes = len(train_df["int_label"].unique())

    # Load in word_index
    word_index = pickle.load(open(f"../../data/interim/{TASK}_word_index.pickle", "rb"))

    # Load in embedding matrix
    embedding_matrix = np.load(f"../../data/interim/{TASK}_embedding_matrix.npy")

    print("\nCreating CNN for Sentence Classification...")

    # Create CNN for sentence classification 
    inputs = Input(shape=(25,))
    conv_layers = SentenceCNN(inputs, word_index, embedding_matrix)
    activation_0 = Activation("relu")(conv_layers)
    dropout_0 = Dropout(0.02)(activation_0)
    dense_0 = Dense(100)(dropout_0)
    activation_1 = Activation("relu")(dense_0)
    dense_1 = Dense(50)(activation_1)
    activation_2 = Activation("relu")(dense_1)
    output_layer = Dense(num_classes, activation="softmax")(activation_2)
    model = Model(inputs=inputs, outputs=output_layer)

    # Initialize Adam optimizer
    adam = Adam(learning_rate=0.01)
    
    # Config model with losses and metrics
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    # Set early-stopping criterion based on the accuracy on the development set with the patience of 10
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, mode="max")

    # Initialize TensorBoard to visualize learning
    tensorboard = TensorBoard(log_dir=f"../../models/text/{TASK}", write_images=True)

    # Create model checkpoints
    checkpoint_filepath = f"../../models/text/{TASK}/{TASK}_checkpoint"
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="max")

    # Train and validate model
    history = model.fit(x=train_X, y=train_y, batch_size=128, epochs=50, validation_data=(val_X, val_y), callbacks=[early_stopping, tensorboard, checkpoint])

    # Load model with best weights
    model.load_weights(checkpoint_filepath)

    # Save trained model with best weights
    model.save(f"../../models/text/{TASK}/{TASK}.hdf5")
