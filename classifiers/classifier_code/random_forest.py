import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import pickle
import joblib
import os
import time
import logging

# Set up logging
log_file = 'classifiers/output/random_forest_output_df_8020.log'
# log_file = 'classifiers/output/random_forest_output_higherRes.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Start time
start_time = time.time()
logging.info("Start Time= %s", time.ctime(start_time))

# # Load the embeddings and multi-labels from separate pickle files
# with open('embeddings/timesformer_video_embeddings.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
#     X = pickle.load(f)
# with open('embeddings/timesformer_video_labels.pkl', 'rb') as f:  # REPLACE LABELS FILE THAT NEEDS TO BE USED
#     y = pickle.load(f)

# Load the DataFrame from the pickle file and Extract features and labels
with open('embeddings/videoMAE_video_embeddings_annotations.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    df = pickle.load(f)
X = df['Video embedding'].apply(lambda x: np.array(x)).tolist()  # Convert list of lists to list of numpy arrays
y = df['Video label'].tolist()

# Convert one-hot encoded labels to multiclass labels for RF only
y = np.argmax(y, axis=1)
y = y.reshape(-1, 1)     #reshaping to column vector

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

#printing first 5 records: video embedding in X and video label in y after reading picklefile
print("Video Embedding:\n", X[:5])
print("Video Labels:\n",y[:5])

# Split into training and testing datasets with an 80:20/90:10 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the splits
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# Ensure X_train and X_test are 2D arrays (flatten if necessary)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

def random_forest_model(X_train, X_test, y_train, y_test):
    # Ensure the number of samples is the same for X and y
    min_samples = min(len(X_train), len(y_train), len(X_test), len(y_test))

    X_train = X_train[:min_samples]
    y_train = y_train[:min_samples]
    X_test = X_test[:min_samples]
    y_test = y_test[:min_samples]

    print("Started Training Random Forest model...")

    # Initialize the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    print("Started Testing the saved Random Forest model...")

    # Predict the test set
    y_pred = model.predict(X_test)

    # Generate confusion matrix for each label and convert to percentage form
    confusion_mats = multilabel_confusion_matrix(y_test, y_pred)

    for i, cm in enumerate(confusion_mats):
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        logging.info('\nConfusion Matrix for Label %d (in %%):\n%s', i, cm_percentage)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logging.info('\nTest Accuracy: %f', accuracy*100)

    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    logging.info('\nWeighted F1 Score: %f', weighted_f1*100)

    # Generate classification report
    class_report = classification_report(y_test, y_pred, digits=6)
    logging.info('\nClassification Report:\n%s', class_report)

    # Print the predicted and actual labels for each video embedding
    for i in range(len(X_test)):
        logging.info('Video %d Embedding:', i+1)
        logging.info('Predicted Labels: %s', y_pred[i])
        logging.info('Actual Labels: %s', y_test[i].numpy())
        logging.info('')

    logging.info("\nSaving the models...")
    
    # Save the model
    model_save_dir = "classifiers/random_forest_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Save as pickle
    pickle_file = os.path.join(model_save_dir, "videoMAE_randomforest.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)
    logging.info('Model saved as pickle file: %s', pickle_file)

    # Save using joblib
    joblib_file = os.path.join(model_save_dir, "videoMAE_randomforest.joblib")
    joblib.dump(model, joblib_file)
    logging.info('Model saved as joblib file: %s', joblib_file)

    # Save as h5 (Note: This will still be a joblib file with .h5 extension)
    h5_file = os.path.join(model_save_dir, "videoMAE_randomforest.h5")
    joblib.dump(model, h5_file)
    logging.info('Model saved as h5 file: %s', h5_file)

# Example usage
random_forest_model(X_train, X_test, y_train, y_test)

# End time
end_time = time.time()
logging.info("\nEnd Time= %s", time.ctime(end_time))
logging.info('Random Forest duration: %f seconds', end_time - start_time)