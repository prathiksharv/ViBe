import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
from itertools import product

device = torch.device("mps")
print("Running on device:", device)

with open('embeddings/videoMAE_video_embeddings_annotations.pkl', 'rb') as f:
    df = pickle.load(f)

X = np.array(df['Video embedding'].tolist())
y = np.array(df['Video label'].tolist())

print("X shape:", X.shape)
print("y shape:", y.shape)

y_indices = np.argmax(y, axis=1)
print("Converted y_indices shape:", y_indices.shape)

D = X.shape[1]
print("Embedding dimension D:", D)

X_train, X_test, y_train_indices, y_test_indices = train_test_split(
    X, y_indices, test_size=0.2)

X_train, X_val, y_train_indices, y_val_indices = train_test_split(
    X_train, y_train_indices, test_size=0.25)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train_indices, dtype=torch.long)
y_val = torch.tensor(y_val_indices, dtype=torch.long)
y_test = torch.tensor(y_test_indices, dtype=torch.long)

print("Train dataset size:", len(X_train))
print("Validation dataset size:", len(X_val))
print("Test dataset size:", len(X_test))

sequence_length_options = [s for s in [ 64, 128, 256, 512, D] if D % s == 0]

hyperparameter_grid = {
    'batch_size': [32, 64],
    'sequence_length': sequence_length_options,
    'hidden_size': [64, 128],
    'num_layers': [1, 2],
    'learning_rate': [0.01, 0.001],
    'optimizer': ['SGD', 'Adam'],
    'num_epochs': [50, 100],
    'model_type': ['RNN', 'LSTM']
}

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, model_type='RNN'):
        super(RNNModel, self).__init__()
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_and_evaluate(hyperparams):
    batch_size = hyperparams['batch_size']
    sequence_length = hyperparams['sequence_length']
    hidden_size = hyperparams['hidden_size']
    num_layers = hyperparams['num_layers']
    learning_rate = hyperparams['learning_rate']
    optimizer_type = hyperparams['optimizer']
    num_epochs = hyperparams['num_epochs']
    model_type = hyperparams['model_type']

    if D % sequence_length != 0:
        return None

    feature_size = D // sequence_length

    def reshape_data(X):
        return X.reshape(X.shape[0], sequence_length, feature_size)

    X_train_seq = reshape_data(X_train.numpy())
    X_val_seq = reshape_data(X_val.numpy())

    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_val_flat = X_val_seq.reshape(X_val_seq.shape[0], -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)

    X_train_scaled = X_train_scaled.reshape(X_train_seq.shape[0], sequence_length, feature_size)
    X_val_scaled = X_val_scaled.reshape(X_val_seq.shape[0], sequence_length, feature_size)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train)
    val_dataset = TensorDataset(X_val_tensor, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = feature_size
    num_classes = len(np.unique(y_indices))

    model = RNNModel(input_size, hidden_size, num_layers, num_classes, model_type).to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

keys, values = zip(*hyperparameter_grid.items())
hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

best_accuracy = 0
best_hyperparams = None

for i, hyperparams in enumerate(hyperparameter_combinations):
    print(f"Hyperparameter combination {i+1}/{len(hyperparameter_combinations)}")
    accuracy = train_and_evaluate(hyperparams)
    if accuracy is not None:
        print(f"Validation Accuracy: {accuracy*100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparams = hyperparams
    else:
        print("Invalid hyperparameter combination (sequence_length does not divide D)")

print("Best validation accuracy:", best_accuracy)
print("Best hyperparameters:", best_hyperparams)

def train_final_model(hyperparams):
    batch_size = hyperparams['batch_size']
    sequence_length = hyperparams['sequence_length']
    hidden_size = hyperparams['hidden_size']
    num_layers = hyperparams['num_layers']
    learning_rate = hyperparams['learning_rate']
    optimizer_type = hyperparams['optimizer']
    num_epochs = hyperparams['num_epochs']
    model_type = hyperparams['model_type']

    if D % sequence_length != 0:
        raise ValueError("Invalid sequence_length")

    feature_size = D // sequence_length

    def reshape_data(X):
        return X.reshape(X.shape[0], sequence_length, feature_size)

    X_combined = torch.cat((X_train, X_val), dim=0)
    y_combined = torch.cat((y_train, y_val), dim=0)

    X_combined_seq = reshape_data(X_combined.numpy())
    X_test_seq = reshape_data(X_test.numpy())

    X_combined_flat = X_combined_seq.reshape(X_combined_seq.shape[0], -1)
    X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    X_combined_scaled = X_combined_scaled.reshape(X_combined_seq.shape[0], sequence_length, feature_size)
    X_test_scaled = X_test_scaled.reshape(X_test_seq.shape[0], sequence_length, feature_size)

    X_combined_tensor = torch.tensor(X_combined_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_combined_tensor, y_combined)
    test_dataset = TensorDataset(X_test_tensor, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = feature_size
    num_classes = len(np.unique(y_indices))

    model = RNNModel(input_size, hidden_size, num_layers, num_classes, model_type).to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print('Test Accuracy:', 100*accuracy)
    print('Weighted F1 Score:', 100*weighted_f1)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=6))

    save_dir = 'RNN_trained'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'RNN_model.pth'))

train_final_model(best_hyperparams)
