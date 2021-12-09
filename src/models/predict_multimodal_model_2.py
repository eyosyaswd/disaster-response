from ast import literal_eval
from custom_dataset import Multimodal_Data_Generator
from performance_metrics import get_performance_metrics
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import concatenate, Dense, MultiHeadAttention, LayerNormalization, Dropout, Embedding, Layer, Input, AveragePooling2D, Flatten, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.models import load_model, Model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Ignore tf info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore tf warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Ignore tf warning messages
import pandas as pd
import pickle
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

    TASK = "informative"

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

    # # Load in trained model (can't save model as hdf5)
    # trained_model = load_model(f"../../models-improved/multimodal/{TASK}/{TASK}.hdf5")

    # print("\nCreating text model...")
    vocab_size = 20000
    print("vocab_size =", vocab_size)

    embed_dim = 300   # Embedding size for each token
    print("embed_dim =", embed_dim)

    maxlen = 25
    print("maxlen =", maxlen)

    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

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

    # print("\nCreating image mode...")

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
    trained_model = Model(inputs=[image_base_model.input, text_inputs], outputs=output_layer)

    # print(trained_model.summary())
    
    # # PC won't let me save model so just recreate the architecture then load in trained weights instead
    # vgg16_model = VGG16(weights="imagenet")
    # fc2 = vgg16_model.get_layer("fc2").output
    # output_layer = Dense(num_classes, activation="softmax")(fc2)
    # model = Model(inputs=vgg16_model.input, outputs=output_layer)
    # checkpoint_filepath = f"../../models-improved/image/{TASK}/{TASK}_checkpoint"
    # trained_model = model.load_weights(checkpoint_filepath)

    print(trained_model.summary()) 

    # Load in weights
    # Load model with best weights
    checkpoint_filepath = f"../../models-improved/multimodal/{TASK}/{TASK}_checkpoint"
    trained_model.load_weights(checkpoint_filepath)

    print("\nPredicting testing data...")
    
    # Predict testing data using trained model
    pred_y = trained_model.predict(test_data_gen, batch_size=1)

    print("\nGetting performance metrics...")

    # Get performance metrics
    get_performance_metrics(test_y, pred_y, test_df, TASK, "multimodal-improved")
    