from custom_dataset import Multimodal_Data_Generator
from sentence_cnn import SentenceCNN
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, concatenate, Activation, GlobalAveragePooling1D, Layer, MultiHeadAttention, LayerNormalization, Embedding, AveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import pickle
import sys
import tensorflow as tf


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
            config = super().get_config().copy()
            config.update({
                'att': self.att,
                'ffn': self.ffn,
                'layernorm1': self.layernorm1,
                'layernorm2': self.layernorm2,
                'dropout1': self.dropout1,
                'dropout2': self.dropout2
            })
            return config


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb
        })
        return config


if __name__ == "__main__":

    TASK = "informative"       # "humanitarian" or "informative"
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
    train_data_gen = Multimodal_Data_Generator(train_df, batch_size=2)
    val_data_gen = Multimodal_Data_Generator(val_df, batch_size=2)

    # Get the number of classes
    num_classes = len(train_df["int_label"].unique())

    print("\nLoading in word_index and embedding_matrix for text model...")

    # Load in word_index
    word_index = pickle.load(open(f"../../data/interim/{TASK}_word_index.pickle", "rb"))
    vocab_size = 20000
    print("vocab_size =", vocab_size)

    # Load in embedding matrix
    embedding_matrix = np.load(f"../../data/interim/{TASK}_embedding_matrix.npy")
    embed_dim = embedding_matrix.shape[1]   # Embedding size for each token
    print("embed_dim =", embed_dim)

    maxlen = 25
    print("maxlen =", maxlen)

    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer


    # NOTE: Can kill the word_index and embedding_matrix variables after passing them 
    # into the text model to free up some space? 

    print("\nCreating text model...")

    text_inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x_text =  embedding_layer(text_inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x_text =  transformer_block(x_text)
    x_text =  GlobalAveragePooling1D()(x_text)
    # x_text =  Dropout(0.2)(x_text)
    x_text =  Dense(500, activation="relu")(x_text)
    batchnorm_text = BatchNormalization()(x_text)
    # x = Dropout(0.2)(x)
    # outputs = Dense(num_classes, activation="softmax")(x)
    # text_model = Model(inputs=text_inputs, outputs=outputs)
    
    # # Create CNN for sentence classification 
    # text_inputs = Input(shape=(25,))
    # conv_layers = SentenceCNN(text_inputs, word_index, embedding_matrix)
    # activation_0 = Activation("relu")(conv_layers)
    # dropout_0 = Dropout(0.02)(activation_0)
    # dense_0_text = Dense(2000, activation="relu")(dropout_0)
    # dense_1_text = Dense(1000, activation="relu")(dense_0_text)
    # batchnorm_text = BatchNormalization()(dense_1_text)

    print("\nCreating image mode...")

    image_base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    image_output_layer = image_base_model.output
    image_output_layer = AveragePooling2D(pool_size=(7, 7))(image_output_layer)
    image_output_layer = Flatten(name="flatten")(image_output_layer)
    image_output_layer = Dense(500, activation="relu")(image_output_layer)
    batchnorm_image = BatchNormalization()(image_output_layer)
    # image_output_layer = Dropout(0.5)(image_output_layer)
    # image_output_layer = Dense(num_classes, activation="softmax")(image_output_layer)

    # Freeze layers of image base model
    for layer in image_base_model.layers:
        layer.trainable = False

    # # Create VGG16 model
    # vgg16_model = VGG16(weights="imagenet")
    # fc2 = vgg16_model.get_layer("fc2").output
    # dense_image = Dense(1000, activation="relu")(fc2)
    # batchnorm_image = BatchNormalization()(dense_image)

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
    model = Model(inputs=[image_base_model.input, text_inputs], outputs=output_layer)

    print(model.summary())
    # sys.exit()

    # Initialize Adam optimizer
    adam = Adam(learning_rate=0.00001)     # NOTE: not specified in paper

    # Config model with losses and metrics
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    # Initialize learning rate reducer
    lr_reducer = ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=5, verbose=1, mode="max")

    # Set early-stopping criterion based on the accuracy on the development set with the patience of 10
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, mode="max")

    # Initialize TensorBoard to visualize learning
    tensorboard = TensorBoard(log_dir=f"../../models-improved/multimodal/{TASK}", write_images=True)

    # Create model checkpoints
    checkpoint_filepath = f"../../models-improved/multimodal/{TASK}/{TASK}_checkpoint"
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="max")

    # Train and validate model
    history = model.fit(x=train_data_gen, epochs=1, validation_data=val_data_gen, callbacks=[lr_reducer, early_stopping, tensorboard, checkpoint])

    # Load model with best weights
    model.load_weights(checkpoint_filepath)

    # Save trained model with best weights
    model.save(f"../../models-improved/multimodal/{TASK}/{TASK}.hdf5")