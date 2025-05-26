import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from sklearn.metrics import confusion_matrix

# ------------------- Data Loading -------------------

# Load the IQ data
iq_data = pkl.load(open('gold.dat', 'rb'), encoding='latin1')
print('Dataset imported')

# Extract modulations and SNRs
snrs, modulation = map(lambda j: sorted(list(set(map(lambda x: x[j], iq_data.keys())))), [1, 0])
print(f'Modulation labels: {modulation}')
print(f'SNR values for each modulation: {snrs}')

# Stack data and labels
x_data = []
labels = []
for m in modulation:
    for snr in snrs:
        samples = iq_data[(m, snr)]
        x_data.append(samples)
        labels += [(m, snr)] * samples.shape[0]

x_stacked = np.vstack(x_data)
print(f'Dataset shape: {x_stacked.shape}')

# ------------------- Train/Test Split -------------------

np.random.seed(200)
N_samples = x_stacked.shape[0]
N_train = int(N_samples * 0.7)
train_idx = np.random.choice(np.arange(N_samples), size=N_train, replace=False)
test_idx = list(set(np.arange(N_samples)) - set(train_idx))

x_train = x_stacked[train_idx]
x_test = x_stacked[test_idx]

print(f'Train shape: {x_train.shape}, Test shape: {x_test.shape}')

# Encode labels
label_encoder = lambda x: modulation.index(labels[x][0])
y_train = np.array(list(map(label_encoder, train_idx)))
y_test = np.array(list(map(label_encoder, test_idx)))

N, H, W = x_train.shape
N_test = x_test.shape[0]
C = 1

x_train = x_train.reshape(N, C, H, W)
x_test = x_test.reshape(N_test, C, H, W)

print(f'x_train: {x_train.shape}, x_test: {x_test.shape}')
print(f'y_train: {y_train.shape}, y_test: {y_test.shape}')

# plt.plot(x_train[0,0,:], x_train[0,1,:], '.')

# ------------------- Dataset Class -------------------

class IQDataset(Dataset):
    def __init__(self, x_data, y_labels):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = IQDataset(x_train, y_train)
test_dataset = IQDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# ------------------- Model Definition -------------------

class CNNArchitecture(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(CNNArchitecture, self).__init__()
        C, H, W = input_shape
        self.pad = nn.ZeroPad2d((2,2,0,0))
        self.conv1 = nn.Conv2d(C, 64, kernel_size=(2,3))
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(64, 80, kernel_size=(1,3))
        self.dropout2 = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            dummy = self.pad(dummy)
            dummy = self.conv1(dummy)
            dummy = self.dropout1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.dropout2(dummy)
            flatten_size = dummy.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pad(x)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------- Training Setup -------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNArchitecture(num_classes=len(modulation), input_shape=(C, H, W)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 10
best_acc = 0.0
counter = 0
save_path = 'cnn_best_model_torch.pth'

# ------------------- Training Loop -------------------

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ------------------- Validation after epoch -------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Check for improvement
    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved new best model at epoch {epoch+1} with accuracy {val_acc:.4f}")
    else:
        counter += 1
        if counter >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

# ------------------- Load best model -------------------
model.load_state_dict(torch.load(save_path))
model.eval()
print(f"Best validation accuracy: {best_acc:.4f}")

# ------------------- Evaluation -------------------

correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:")
# print(cm)
