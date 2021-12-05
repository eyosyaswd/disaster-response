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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


if __name__ == "__main__":

    TASK = "informative"       # "humanitarian" or "informative"
    SEED = 2021                # Seed to be used for reproducability

    print("\nLoading in dataset, word_index, and embedding_matrix...")

    # Specify location of csv files containing preprocessed data
    train_filepath = f"../../data/interim/task_{TASK}_train_preprocessed_text.csv"
    val_filepath = f"../../data/interim/task_{TASK}_val_preprocessed_text.csv"

    # Load in csv files containing preprocessed data
    train_df = pd.read_csv(train_filepath)
    val_df = pd.read_csv(val_filepath)

    # Extract data and labels from dataset
    train_X = list(train_df["padded_sequence"].apply(literal_eval))
    train_y = list(train_df["onehot_label"].apply(literal_eval))
    val_X = list(val_df["padded_sequence"].apply(literal_eval))
    val_y = list(val_df["onehot_label"].apply(literal_eval))

    # Get the number of classes
    num_classes = len(train_df["int_label"].unique())

    # Load in word_index
    word_index = pickle.load(open(f"../../data/interim/{TASK}_word_index.pickle", "rb"))
    vocab_size = len(word_index)    # Only consider this many words
    print("vocab_size =", vocab_size)

    # Load in embedding matrix
    embedding_matrix = np.load(f"../../data/interim/{TASK}_embedding_matrix.npy")
    embed_dim = embedding_matrix.shape[1]   # Embedding size for each token
    print("embed_dim =", embed_dim)

    maxlen = 25
    print("maxlen =", maxlen)

    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer


    print("\nCreating Transformer for Sentence Classification...")

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # # Create CNN for sentence classification 
    # inputs = Input(shape=(25,))
    # conv_layers = SentenceCNN(inputs, word_index, embedding_matrix)
    # activation_0 = Activation("relu")(conv_layers)
    # dropout_0 = Dropout(0.02)(activation_0)
    # dense_0 = Dense(100, activation="relu")(dropout_0)
    # dense_1 = Dense(50, activation="relu")(dense_0)
    # output_layer = Dense(num_classes, activation="softmax")(dense_1)
    # model = Model(inputs=inputs, outputs=output_layer)

    print(model.summary())
    # sys.exit()

    # Initialize Adam optimizer
    adam = Adam(learning_rate=0.01)
    
    # Config model with losses and metrics
    # model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])


    # Set early-stopping criterion based on the accuracy on the development set with the patience of 10
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, mode="max")

    # Initialize TensorBoard to visualize learning
    tensorboard = TensorBoard(log_dir=f"../../models-improved/text/{TASK}", write_images=True)

    # Create model checkpoints
    checkpoint_filepath = f"../../models-improved/text/{TASK}/{TASK}_checkpoint"
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="max")

    # Train and validate model
    history = model.fit(x=train_X, y=train_y, batch_size=32, epochs=2, validation_data=(val_X, val_y), callbacks=[early_stopping, tensorboard, checkpoint])

    # Load model with best weights
    model.load_weights(checkpoint_filepath)

    # Save trained model with best weights
    model.save(f"../../models-improved/text/{TASK}/{TASK}.hdf5")
