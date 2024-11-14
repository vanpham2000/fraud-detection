import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle

# Load and prepare data
data_path = 'data source.csv'  # Path to your data file
data = pd.read_csv(data_path)

# Encode categorical data
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

# Select features and target
features = data[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
target = data['isFraud']

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Create dataloaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define the neural network
class FraudNet(nn.Module):
    def __init__(self):
        super(FraudNet, self).__init__()
        self.fc1 = nn.Linear(7, 64)  # 7 features as input
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Initialize the model, loss criterion, and optimizer
model = FraudNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# Evaluate the model
def evaluate_model():
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')


# Run training and evaluation
train_model(10)
evaluate_model()

# Save the trained model
torch.save(model.state_dict(), 'fraud_detection_model.pth')

# Save scaler and label encoder
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model, scaler, and label encoder saved successfully.")
