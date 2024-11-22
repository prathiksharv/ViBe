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


if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")
else:
    device = torch.device("mps")
print("Running on device:", device)


with open('embeddings/videoMAE_video_embeddings_annotations.pkl', 'rb') as f:
    df = pickle.load(f)

X = df['Video embedding'].tolist()  
y = df['Video label'].tolist()      


X = np.array(X)  
y = np.array(y)  

print("X shape:", X.shape)
print("y shape:", y.shape)


y_indices = np.argmax(y, axis=1)
print("Converted y_indices shape:", y_indices.shape)


D = X.shape[1]
print("Embedding dimension D:", D)


sequence_length = 8  
if D % sequence_length != 0:
    raise ValueError("Embedding dimension D is not divisible by sequence_length")

feature_size = D // sequence_length


X = X.reshape(X.shape[0], sequence_length, feature_size)
print("Reshaped X shape:", X.shape)


X = X[:, np.newaxis, :, :]
print("X reshaped for CNN input:", X.shape)


X_train, X_test, y_train_indices, y_test_indices = train_test_split(
    X, y_indices, test_size=0.2, random_state=42)


X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)


X_train_scaled = X_train_scaled.reshape(
    X_train.shape[0], 1, sequence_length, feature_size)
X_test_scaled = X_test_scaled.reshape(
    X_test.shape[0], 1, sequence_length, feature_size)


X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_indices, dtype=torch.long)
y_test = torch.tensor(y_test_indices, dtype=torch.long)


batch_size = 128
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.flatten_size = 64 * (sequence_length // 2) * (feature_size // 2)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  
        x = torch.relu(self.conv2(x))  
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  
        return x


num_classes = len(np.unique(y_indices))
model = SimpleCNN(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


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
print('Test Accuracy:', accuracy * 100)

weighted_f1 = f1_score(y_true, y_pred, average='weighted')
print('Weighted F1 Score:', weighted_f1 * 100)

print('Classification Report:')
print(classification_report(y_true, y_pred, digits=6))


save_dir = 'CNN_trained'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model.state_dict(), os.path.join(save_dir, 'CNN_model.pth'))
