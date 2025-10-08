import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the data
data = np.load("data.npy", allow_pickle=True)
df = pd.DataFrame(data, columns=["Edge Code", "Type", "Label"])

# Map types to integers for classification
type_mapping = {type_name: idx for idx, type_name in enumerate(df["Type"].unique())}
df["Type"] = df["Type"].map(type_mapping)

X = df["Edge Code"].astype(str)
y_label = df["Label"].astype(float)
y_type = df["Type"].astype(int)

X_train, X_test, y_label_train, y_label_test, y_type_train, y_type_test = train_test_split(
    X, y_label, y_type, test_size=0.7, random_state=42
)

# Preprocess sequences
def preprocess_sequences(bitstrings):
    sequences = []
    for bitstring in bitstrings:
        sequence = []
        for bit in bitstring:
            sequence.append(int(bit))
        sequences.append(torch.tensor(sequence, dtype=torch.float32))
    return sequences

train_sequences = preprocess_sequences(X_train)
test_sequences = preprocess_sequences(X_test)

train_label_targets = torch.tensor(y_label_train.values, dtype=torch.float32)
train_type_targets = torch.tensor(y_type_train.values, dtype=torch.long)

test_label_targets = torch.tensor(y_label_test.values, dtype=torch.float32)
test_type_targets = torch.tensor(y_type_test.values, dtype=torch.long)

# Custom Dataset class
class BitstringDataset(Dataset):
    def __init__(self, sequences, label_targets, type_targets):
        self.sequences = sequences
        self.label_targets = label_targets
        self.type_targets = type_targets

    def __len__(self):
        return len(self.label_targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.label_targets[idx], self.type_targets[idx]

train_dataset = BitstringDataset(train_sequences, train_label_targets, train_type_targets)
test_dataset = BitstringDataset(test_sequences, test_label_targets, test_type_targets)

# DataLoader
def collate_fn(batch):
    sequences, label_targets, type_targets = zip(*batch)
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True).unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
    label_targets = torch.tensor(label_targets, dtype=torch.float32).view(-1, 1)
    type_targets = torch.tensor(type_targets, dtype=torch.long)
    return sequences, label_targets, type_targets

train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False, collate_fn=collate_fn)

# RNN Classifier
class RNNMultiTaskClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_label_size, output_type_size, num_layers=1):
        super(RNNMultiTaskClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_label = nn.Linear(hidden_size, output_label_size)  # Binary classification
        self.fc_type = nn.Linear(hidden_size, output_type_size)   # Multi-class classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Output from the last time step

        # Separate heads for label and type
        label_out = self.sigmoid(self.fc_label(out))
        type_out = self.fc_type(out)  # Raw logits for type classification
        return label_out, type_out

# Model hyperparameters
input_size = 1
hidden_size = 128
output_label_size = 1
output_type_size = len(type_mapping)
num_epochs = 1000
learning_rate = 0.01

# Initialize model, loss functions, optimizer, and scheduler
model = RNNMultiTaskClassifier(input_size, hidden_size, output_label_size, output_type_size)
criterion_label = nn.BCELoss()  # For binary classification
criterion_type = nn.CrossEntropyLoss()  # For type classification
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# Training with early stopping
best_loss = float('inf')
patience = 10
no_improve_epochs = 0
best_model_state = None

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for sequences, label_targets, type_targets in train_dataloader:
        # Forward pass
        label_outputs, type_outputs = model(sequences)

        # Compute losses
        loss_label = criterion_label(label_outputs, label_targets)
        loss_type = criterion_type(type_outputs, type_targets)
        loss = loss_label + loss_type  # Combine losses
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)

    # Update the learning rate scheduler
    scheduler.step(avg_loss)

    # Print the current learning rate
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

    # Check for improvement
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = model.state_dict()
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # Early stopping
    if no_improve_epochs >= patience:
        print("Early stopping triggered.")
        break


# Save the best model
torch.save(best_model_state, "best_model.pth")

# Evaluation
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    correct_labels = 0
    correct_types = 0
    total = 0
    for sequences, label_targets, type_targets in test_dataloader:
        label_outputs, type_outputs = model(sequences)

        # Predictions
        label_preds = (label_outputs > 0.5).float()
        type_preds = torch.argmax(type_outputs, dim=1)

        correct_labels += (label_preds.view(-1) == label_targets.view(-1)).sum().item()
        correct_types += (type_preds == type_targets).sum().item()
        total += label_targets.size(0)

    label_accuracy = 100 * correct_labels / total
    type_accuracy = 100 * correct_types / total
    print(f"Label Accuracy: {label_accuracy:.2f}%")
    print(f"Type Accuracy: {type_accuracy:.2f}%")
