import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
import os
import time
import logging
import joblib

# Set up logging
log_file = 'classifiers/output/gru_output_df_8020_Newbatch_and_Epoch.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Start time
start_time = time.time()
logging.info("Start Time= %s", time.ctime(start_time))
    
# Load the DataFrame from the pickle file and extract features and labels
with open('embeddings/videoMAE_video_embeddings_annotations.pkl', 'rb') as f:  # REPLACE EMBEDDING FILE THAT NEEDS TO BE USED
    df = pickle.load(f)
X = df['Video embedding'].apply(lambda x: np.array(x)).tolist()  # Convert list of lists to list of numpy arrays
y = df['Video label'].tolist()

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert one-hot encoded labels to integer class labels
y = np.argmax(y, axis=1)
y = y.reshape(-1, 1)     #reshaping to column vector

# Printing first 5 records: video embedding in X and video label in y after reading picklefile
logging.info("Video Embedding:\n%s", X[:5])
logging.info("Video Labels:\n%s", y[:5])

num_classes = 7

# Split into training and testing datasets with an 80:20/90:10 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the sizes of the splits
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

def gru_model(X_train, X_test, y_train, y_test, epochs):
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the input data to include the batch size
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    logging.info("Started Training GRU model...")

    # Build the GRU model
    model = Sequential()
    model.add(tf.keras.Input(shape=(1, X_train_scaled.shape[2])))
    model.add(GRU(units=64, return_sequences=True, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(GRU(units=64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    # Additional Dense layers
    model.add(Dense(units=25, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(units=15, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units=10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model with a lower learning rate
    optimizer = AdamW(learning_rate=1e-5, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Train the model with early stopping and learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20))
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=128, validation_data=(X_test_scaled, y_test), callbacks=[lr_scheduler])

    logging.info("Started Testing the saved GRU model...")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    logging.info('Test Loss: %f, Test Accuracy: %f', loss, accuracy*100)

    # Print Classification Report
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    report = classification_report(y_test, y_pred, digits=6)
    logging.info('\nClassification Report:\n%s', report)
    
    # Print the predicted and actual labels for each video embedding
    for i in range(len(X_test)):
        logging.info('Video %d Embedding:', i+1)
        logging.info('Predicted Labels: %s', y_pred[i])
        logging.info('Actual Labels: %s', y_test[i])
        logging.info('')

    logging.info("\nSaving the models...")
    
    # Save the model in multiple formats
    model_save_dir = "classifiers/gru_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Save as .h5 file
    h5_file = os.path.join(model_save_dir, "videoMAE_gru_model_82.h5")
    model.save(h5_file)
    logging.info('Model saved as h5 file: %s', h5_file)

    # Save as .keras file
    keras_file = os.path.join(model_save_dir, "videoMAE_gru_model_82.keras")
    model.save(keras_file)
    logging.info('Model saved as keras file: %s', keras_file)
    
    # Save as .pickle file
    pickle_file = os.path.join(model_save_dir, "videoMAE_gru_model_82.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)
    logging.info('Model saved as pickle file: %s', pickle_file)

    # Save using joblib
    joblib_file = os.path.join(model_save_dir, "videoMAE_gru_model_82.joblib")
    joblib.dump(model, joblib_file)
    logging.info('Model saved as joblib file: %s', joblib_file)


# Example usage
epochs = 100
gru_model(X_train, X_test, y_train, y_test, epochs)

# End time
end_time = time.time()
logging.info("\nEnd Time= %s", time.ctime(end_time))
logging.info('GRU duration: %f seconds', end_time - start_time)
