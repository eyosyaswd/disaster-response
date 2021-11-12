from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pickle


def get_performance_metrics(true_y, pred_y, test_df, task, modality):
    # Get the index (aka int_label) with highest probability
    y_true = np.argmax(true_y, axis=1)
    y_pred = np.argmax(pred_y, axis=1)

    # Read in label_encoder preduced from training data
    le = pickle.load(open(f"../../data/interim/{task}_label_encoder.pickle", "rb"))

    # Convert labels from ints to string labels 
    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)

    # Get performance metrics
    class_report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred, average="weighted")
    rec_score = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nClassification Report:\n{class_report}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print("\nAccuracy Score:", acc_score)
    print("\nPrecision Score:", prec_score)
    print("\nRecall Score:", rec_score)
    print("\nF1 Score:", f1)

    performance_file = open(f"../../reports/task={task},modality={modality}.txt", "w")
    performance_file.write(f"\nClassification Report:\n{class_report}")
    performance_file.write(f"\nConfusion Matrix:\n{conf_matrix}")
    performance_file.write(f"\n\nAccuracy Score: {acc_score}")
    performance_file.write(f"\n\nPrecision Score: {prec_score}")
    performance_file.write(f"\n\nRecall Score: {rec_score}")
    performance_file.write(f"\n\nF1 Score: {f1}")
    performance_file.close()

    test_df[f"{modality}_prediction"] = y_pred
    test_df.to_csv(f"../../data/processed/task={task},modality={modality},predictions.csv", index=False)
