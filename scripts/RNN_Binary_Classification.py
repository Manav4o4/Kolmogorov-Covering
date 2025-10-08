import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

data = np.load('data.npy', allow_pickle=True)

df = pd.DataFrame(data, columns=['Edge Code', 'Type', 'Label'])

X = df['Edge Code']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7)

train_sequences = [torch.tensor([int(bit) for bit in bitstring], dtype=torch.float32) for bitstring in X_train]
train_labels = torch.tensor([label for label in y_train], dtype=torch.float32)

test_sequences = [torch.tensor([int(bit) for bit in bitstring], dtype=torch.float32) for bitstring in X_test]
test_labels = torch.tensor([label for label in y_test], dtype=torch.float32)

class BitstringDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = BitstringDataset(train_sequences, train_labels)
test_dataset = BitstringDataset(test_sequences, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=50, collate_fn=lambda x: collate_fn(x))
test_dataloader = DataLoader(test_dataset, batch_size=50, collate_fn=lambda x: collate_fn(x))

def collate_fn(batch):

    sequences, labels = zip(*batch)
    sequences = torch.tensor(labels, dtype=torch.float32).view(-1,1)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return sequences, labels

import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Adjust for num_layers
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Take the output of the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through a fully connected layer
        out = self.fc(out)
        return self.sigmoid(out)
    
# Hyperparameters
input_size = 1       # Each bit (0 or 1) is treated as one feature
hidden_size = 128    # Number of units in LSTM hidden layer
output_size = 1      # Output for binary classification (0 or 1)
num_epochs = 200
learning_rate = 0.001

# Initialize model, criterion, and optimizer
model = RNNClassifier(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for sequences, labels in train_dataloader:
        # Reshape sequences to add the input dimension
        sequences = sequences.unsqueeze(-1)  # (batch_size, seq_length, input_size)

        labels = labels.view(-1, 1)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation mode
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for sequences, labels in test_dataloader:
        sequences = sequences.unsqueeze(-1)
        outputs = model(sequences)
        predictions = (outputs > 0.95).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')