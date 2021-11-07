from ast import literal_eval
from performance_metrics import get_performance_metrics
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Ignore tf info messages
import pandas as pd


if __name__ == "__main__":

    TASK = "humanitarian"

    print("\nLoading in testing data...")

    # Read in testing data
    test_filepath = f"../../data/interim/task_{TASK}_test_preprocessed_text.csv"
    test_df = pd.read_csv(test_filepath)

    # Extract data and labels from dataset
    test_X = list(test_df["padded_sequence"].apply(literal_eval))
    test_y = list(test_df["onehot_label"].apply(literal_eval))

    print("\nLoading in trained model...")

    # Load in trained model 
    trained_model = load_model(f"../../models/text/{TASK}/{TASK}.hdf5")

    print(trained_model.summary()) 

    print("\nPredicting testing data...")
    
    # Predict testing data using trained model
    pred_y = trained_model.predict(test_X, batch_size=128)

    print("\nGetting performance metrics...")

    # Get performance metrics
    get_performance_metrics(test_y, pred_y, test_df)
    