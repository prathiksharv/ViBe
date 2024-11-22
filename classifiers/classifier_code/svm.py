import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import joblib
import os
import time
import logging

# Set up logging
log_file = 'classifiers/output/svm_output_df_9010.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Start time
start_time = time.time()
logging.info("Start Time= %s", time.ctime(start_time))
    
# Load the DataFrame from the pickle file and Extract features and labels
with open('embeddings/videoMAE_video_embeddings_annotations.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    df = pickle.load(f)
X = df['Video embedding'].apply(lambda x: np.array(x)).tolist()  # Convert list of lists to list of numpy arrays
y = df['Video label'].tolist()
print("y",y)

# Convert one-hot encoded labels to multiclass labels
y_multiclass = np.argmax(y, axis=1)
print("y_multiclass",y_multiclass)

# Convert lists to numpy arrays
X = np.array(X)
y_multiclass = np.array(y_multiclass).reshape(-1, 1)  # Reshape to column vector

# Printing first 5 records: video embedding in X and video label in y after reading pickle file
logging.info("Video Embedding:\n%s", X[:5])
logging.info("Video Labels:\n%s", y_multiclass[:5])

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.1, random_state=42)

# Print the sizes of the splits
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# Ensure X_train and X_test are 2D arrays (flatten if necessary)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

print("Started Training SVM model...")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Support Vector Classifier model
svc = SVC(kernel='linear', probability=True, random_state=42)

# Train the model
svc.fit(X_train_scaled, y_train.ravel())  # Flatten y_train for training

print("Started Testing the saved SVM model...")

# Predict the test set
y_pred = svc.predict(X_test_scaled).reshape(-1, 1)  # Reshape prediction to column vector

# Generate confusion matrix and convert to percentage form
cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
logging.info('\nConfusion Matrix (in %%):\n%s', cm_percentage)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
logging.info('\nTest Accuracy: %f', accuracy * 100)

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
logging.info('\nWeighted F1 Score: %f', weighted_f1 * 100)

# Generate classification report
class_report = classification_report(y_test, y_pred, digits=6)
logging.info('\nClassification Report:\n%s', class_report)

# Print the predicted and actual labels for each video embedding
for i in range(len(X_test)):
    logging.info('Video %d Embedding:', i + 1)
    logging.info('Predicted Labels: %s', y_pred[i])
    logging.info('Actual Labels: %s', y_test[i])
    logging.info('')

print("\nSaving the models...")
# Save the model
model_save_dir = "classifiers/svm_models"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Save as pickle
pickle_file = os.path.join(model_save_dir, "videoMAE_svm.pkl")
with open(pickle_file, 'wb') as f:
    pickle.dump(svc, f)
logging.info('Model saved as pickle file: %s', pickle_file)

# Save using joblib
joblib_file = os.path.join(model_save_dir, "videoMAE_svm.joblib")
joblib.dump(svc, joblib_file)
logging.info('Model saved as joblib file: %s', joblib_file)

# Save as h5 (Note: This will still be a joblib file with .h5 extension)
h5_file = os.path.join(model_save_dir, "videoMAE_svm.h5")
joblib.dump(svc, h5_file)
logging.info('Model saved as h5 file: %s', h5_file)

# End time
end_time = time.time()
logging.info("\nEnd Time= %s", time.ctime(end_time))
logging.info('SVM duration: %f seconds', end_time - start_time)

