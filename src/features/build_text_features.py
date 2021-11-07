from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from scipy import sparse
from scipy.sparse import data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

# Load in stop words
nltk.download("stopwords")
stop_words = stopwords.words("english")


def preprocess_text(text):
    """ Given a piece of text (a tweet), preprocesses text according to paper.
    
    Text preprocessing steps:
    - Remove stop words
    - Remove non-ASCII characters
    - Remove numbers
    - Remove URLs
    - Remove hashtags
    - Replace all punctuation marks with white-spaces

    NOTE: I would do more preprocessing (e.g. remove 'RT' and the mentioned account in retweets)
          but this is the only thing mentioned in the paper.
    """

    # Convert text to lowercase
    cleaned_text = text.lower()

    # Remove non-ASCII characters 
    cleaned_text = cleaned_text.encode(encoding="ascii", errors="ignore").decode()

    # Remove numbers
    cleaned_text = re.sub(r"[0-9]", "", cleaned_text)

    # Remove URLs
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)

    # Remove hashtag signs
    cleaned_text = cleaned_text.replace("#", "")

    # Replace all punctuation marks with white-spaces
    cleaned_text = re.sub(r"[,.:;@#?!&$]+\ *", " ", cleaned_text)

    # Remove stop words
    tokenized_text = TweetTokenizer().tokenize(cleaned_text)
    cleaned_text = " ".join([token for token in tokenized_text if token not in stop_words])

    return cleaned_text


def preprocess_data(df, data_type, le=None, ohe=None, tokenizer=None):
    """ Preprocess the dataset passed in as a dataframe. 
    
    Preprocessing steps:
    - Clean the text
    - Create label encodings 
    - Tokenize the text (as a sequence of integers, rather than sequence of words)
    - Pad the tokenized sequence
    """

    # Clean text data
    df["preprocessed_text"] = df["tweet_text"].apply(lambda x: preprocess_text(str(x)))

    # Create label encoding for dataset; OneHotEncoder() can't take in strings so 
    # first do integer encoding then convert integer encoding into one-hot encoding
    if data_type == "train": 
        # if training data, create encodings from scratch (fit and transform)
        le, ohe = LabelEncoder(), OneHotEncoder(sparse=False)
        df["int_label"] = le.fit_transform(df["label"])
        df["onehot_label"] = ohe.fit_transform(np.array(df["int_label"]).reshape(-1,1)).tolist()
    else:
        # if val or test data, use the encoders that were fit on training data (only transform)
        df["int_label"] = le.transform(df["label"])
        df["onehot_label"] = ohe.transform(np.array(df["int_label"]).reshape(-1,1)).tolist()

    # Fit tokenizer
    if data_type == "train":
        # if training data, create new tokenizer object and fit training data on it
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token="OOV_TOK")
        tokenizer.fit_on_texts(df["preprocessed_text"])

    # Convert texts to sequences (tokenize)
    # sequence = list of indices that represent the index of the word in the corpus (tokenizer.word_index)
    # the length of the sequence equals the length of the tweet 
    sequences = tokenizer.texts_to_sequences(df["preprocessed_text"])
    
    # Zero-pad the sequences
    df["padded_sequence"] = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=25).tolist()

    # Shuffle data
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df, le, ohe, tokenizer


def generate_embedding_matrix(word2vec_model, word_index):
    """ Generate an embedding matrix (list of word embeddings) for all the words in word_index. """

    # Initialize empty embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, 300), dtype=np.float32)
    
    # Iterate through words and their values (indices)
    for word, idx in word_index.items():
        try:
            # Get its embedding for the word if it has previosuly been seen by the word2vec_model
            embedding_matrix[idx] = word2vec_model[word][0:300]
        except KeyError:
            # Generate a random word embedding (using Normal dist) if the word is not in the model
            embedding_matrix[idx] = np.random.RandomState().randn(300)
    
    return embedding_matrix


if __name__ == "__main__":

    TASK = "humanitarian"       # "humanitarian" or "informative"
    SEED = 2021                    # Seed to be used for reproducability
    GEN_EMBEDDING_MATRIX = False

    print("\nLoading in data...")

    # Specify location of tsv files containing dataset
    train_filepath = f"../../data/raw/crisismmd_datasplit_agreed_label/task_{TASK}_text_img_agreed_lab_train.tsv"
    val_filepath = f"../../data/raw/crisismmd_datasplit_agreed_label/task_{TASK}_text_img_agreed_lab_dev.tsv"
    test_filepath = f"../../data/raw/crisismmd_datasplit_agreed_label/task_{TASK}_text_img_agreed_lab_test.tsv"

    # Load in tsv files containing data
    train_df = pd.read_csv(train_filepath, sep="\t")
    val_df = pd.read_csv(val_filepath, sep="\t")
    test_df = pd.read_csv(test_filepath, sep="\t")

    print("\nPreprocessing data...")

    # Preprocess data
    train_df, le, ohe, tokenizer = preprocess_data(train_df, data_type="train")
    val_df, _, _, _ = preprocess_data(val_df, data_type="val", le=le, ohe=ohe, tokenizer=tokenizer)
    test_df, _, _, _ = preprocess_data(test_df, data_type="test", le=le, ohe=ohe, tokenizer=tokenizer)

    # Save label_encoder (to be used during testing later)
    pickle.dump(le, open("../../data/interim/label_encoder.pickle", "wb"))

    # Output preprocessed data
    train_df.to_csv(f"../../data/interim/task_{TASK}_train_preprocessed_text.csv", index=False)
    val_df.to_csv(f"../../data/interim/task_{TASK}_val_preprocessed_text.csv", index=False)
    test_df.to_csv(f"../../data/interim/task_{TASK}_test_preprocessed_text.csv", index=False)

    print("\nCreating embeddings...")

    # Get word_index dictionary which contains the training data corpus 
    # key: word, value: int b/t 1 and len(num of words in corpus) corresponding to word frequency (lower = more frequent) 
    word_index = tokenizer.word_index

    # Save word_index
    pickle.dump(word_index, open("../../data/interim/word_index.pickle", "wb"))
    
    # Generate or read in embedding matrix 
    # embedding_matrix = MxN matrix, M = number of words in training corpus (len(word_index)), N = size of word2vec embeddings (300)
    if GEN_EMBEDDING_MATRIX is True:
        # Load in word2vec model
        word2vec_model = KeyedVectors.load_word2vec_format('../../data/raw/crisisNLP_word2vec_model/crisisNLP_word_vector.bin', binary=True)
    
        # Generate word embeddings for words in word_index using the word2vec model
        embedding_matrix = generate_embedding_matrix(word2vec_model, word_index)

        # Save embedding matrix
        np.save("../../data/interim/embedding_matrix.npy", embedding_matrix)
    else:
        embedding_matrix = np.load("../../data/interim/embedding_matrix.npy")

