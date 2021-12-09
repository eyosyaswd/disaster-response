from ast import literal_eval
from performance_metrics import get_performance_metrics
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Ignore tf info messages
import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D, Layer, MultiHeadAttention, LayerNormalization, Embedding
from tensorflow.keras.models import Model
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

    TASK = "humanitarian"

    print("\nLoading in testing data...")

    # Read in testing data
    test_filepath = f"../../data/interim/task_{TASK}_test_preprocessed_text.csv"
    test_df = pd.read_csv(test_filepath)

    # Extract data and labels from dataset
    test_X = list(test_df["padded_sequence"].apply(literal_eval))
    test_y = list(test_df["onehot_label"].apply(literal_eval))

    # Get the number of classes
    num_classes = len(test_df["int_label"].unique())
    vocab_size = 20000
    embed_dim = 300   # Embedding size for each token
    maxlen = 25
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    print("\nLoading in trained model...")
    # print("\nCreating Transformer for Sentence Classification...")

    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    # x = Dropout(0.2)(x)
    x = Dense(20, activation="relu")(x)
    # x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)

    # Load in trained model 
    checkpoint_filepath = f"../../models-improved/text/{TASK}/{TASK}_checkpoint"
    model.load_weights(checkpoint_filepath)

    print(model.summary()) 

    print("\nPredicting testing data...")
    
    # Predict testing data using trained model
    pred_y = model.predict(test_X, batch_size=128)

    print("\nGetting performance metrics...")

    # Get performance metrics
    get_performance_metrics(test_y, pred_y, test_df, TASK, "text_transformer")
    