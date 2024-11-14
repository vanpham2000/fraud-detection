from flask import Flask, request, jsonify
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from data import FraudNet  # Import your model class here

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = FraudNet()
model.load_state_dict(torch.load('fraud_detection_model.pth'))
model.eval()

# Load the scaler and label encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_data = pd.DataFrame(data)

    # Preprocess the new data
    new_data['type'] = label_encoder.transform(new_data['type'])
    new_features = new_data[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    new_features = scaler.transform(new_features)
    new_features = torch.tensor(new_features, dtype=torch.float32)

    # Perform prediction
    with torch.no_grad():
        outputs = model(new_features).squeeze()
        predictions = (outputs > 0.5).float().numpy()  # Binary prediction threshold of 0.5

    # Return predictions as JSON
    return jsonify(predictions.tolist())


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
