from custom_dataset import Multimodal_Data_Generator
from sentence_cnn import SentenceCNN
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import pickle


if __name__ == "__main__":

    TASK = "humanitarian"       # "humanitarian" or "informative"
    SEED = 2021                # Seed to be used for reproducability

    print("\nLoading in training and validation datasets...")

    # Specify location of csv files containing training and validation data
    train_filepath = f"../../data/interim/task_{TASK}_train_preprocessed_text.csv"
    val_filepath = f"../../data/interim/task_{TASK}_val_preprocessed_text.csv"

    # Load in csv files containing training and validation data
    train_df = pd.read_csv(train_filepath)
    val_df = pd.read_csv(val_filepath)

    # NOTE: we're not going to load the actual datasets here. Too many OOM errors.
    # Instead, we're going to load the data into memory during training using data generators. 
    train_data_gen = Multimodal_Data_Generator(train_df, batch_size=16)
    val_data_gen = Multimodal_Data_Generator(val_df, batch_size=16)

    # Get the number of classes
    num_classes = len(train_df["int_label"].unique())

    print("\nLoading in word_index and embedding_matrix for text model...")

    # Load in word_index
    word_index = pickle.load(open(f"../../data/interim/{TASK}_word_index.pickle", "rb"))

    # Load in embedding matrix
    embedding_matrix = np.load(f"../../data/interim/{TASK}_embedding_matrix.npy")

    # NOTE: Can kill the word_index and embedding_matrix variables after passing them 
    # into the text model to free up some space? 

    print("\nCreating text model...")
    
    # Create CNN for sentence classification 
    text_inputs = Input(shape=(25,))
    conv_layers = SentenceCNN(text_inputs, word_index, embedding_matrix)
    dense_text = Dense(1000, activation="relu")(conv_layers)
    batchnorm_text = BatchNormalization()(dense_text)

    print("\nCreating image mode...")

    # Create VGG16 model
    vgg16_model = VGG16(weights="imagenet")
    fc2 = vgg16_model.get_layer("fc2").output
    dense_image = Dense(1000, activation="relu")(fc2)
    batchnorm_image = BatchNormalization()(dense_image)

    print("\nCreating text and image merged model...")

    # Concatenate last layers of both models
    concat_multimodal = concatenate([batchnorm_image, batchnorm_text])
    batchnorm_multimodal = BatchNormalization()(concat_multimodal)
    dropout_0_multimodal = Dropout(0.4)(batchnorm_multimodal)
    dense_0_multimodal = Dense(500, activation="relu")(dropout_0_multimodal)
    dropout_1_multimodal = Dropout(0.2)(dense_0_multimodal)
    dense_1_multimodal = Dense(100, activation="relu")(dropout_1_multimodal)
    dropout_2_multimodal = Dropout(0.02)(dense_1_multimodal)
    output_layer = Dense(num_classes, activation="softmax")(dropout_2_multimodal)
    model = Model(inputs=[vgg16_model.input, text_inputs], outputs=output_layer)

    print(model.summary())

    # Initialize Adam optimizer
    adam = Adam(learning_rate=0.00001)     # NOTE: not specified in paper

    # Config model with losses and metrics
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    # Initialize learning rate reducer
    lr_reducer = ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=5, verbose=1, mode="max")

    # Set early-stopping criterion based on the accuracy on the development set with the patience of 10
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, mode="max")

    # Initialize TensorBoard to visualize learning
    tensorboard = TensorBoard(log_dir=f"../../models/multimodal/{TASK}", write_images=True)

    # Create model checkpoints
    checkpoint_filepath = f"../../models/multimodal/{TASK}/{TASK}_checkpoint"
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="max")

    # Train and validate model
    history = model.fit(x=train_data_gen, epochs=50, validation_data=val_data_gen, callbacks=[lr_reducer, early_stopping, tensorboard, checkpoint])

    # Load model with best weights
    model.load_weights(checkpoint_filepath)

    # Save trained model with best weights
    model.save(f"../../models/multimodal/{TASK}/{TASK}.hdf5")