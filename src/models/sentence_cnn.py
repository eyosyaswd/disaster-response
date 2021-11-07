from tensorflow.keras.layers import Embedding, Reshape, Conv2D, MaxPool2D, concatenate, Flatten


def SentenceCNN(inputs, word_index, embedding_matrix):
    """
    Implementation of CNN for Sentence Classification by Yoon Kim.
    Source: https://www.kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim

    """

    embedding_dim = 300
    num_filters = [100, 150, 200]
    sequence_length = 25

    # Embedding layer
    embedding_layer = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, input_length=sequence_length, weights=[embedding_matrix], trainable=True)(inputs)

    # Reshape
    reshape = Reshape(target_shape=(sequence_length, embedding_dim, 1))(embedding_layer)

    # Convolution windows
    conv_0 = Conv2D(filters=num_filters[0], kernel_size=(2, embedding_dim), padding="valid", kernel_initializer="normal", activation="relu")(reshape)
    conv_1 = Conv2D(filters=num_filters[1], kernel_size=(3, embedding_dim), padding="valid", kernel_initializer="normal", activation="relu")(reshape)
    conv_2 = Conv2D(filters=num_filters[2], kernel_size=(4, embedding_dim), padding="valid", kernel_initializer="normal", activation="relu")(reshape)
    
    # Perform max pooling on each of the convolutions
    maxpool_0 = MaxPool2D(pool_size=(sequence_length - 2 + 1, 1), strides=(1,1), padding="valid")(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - 3 + 1, 1), strides=(1,1), padding="valid")(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - 4 + 1, 1), strides=(1,1), padding="valid")(conv_2)

    # Concatenate and flatten 
    concatenated_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2])

    flatten = Flatten()(concatenated_tensor)

    return flatten