import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
from utils.image_processing import preprocess_image
from database.db import save_patient_data

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load pre-trained LSTM model
model = LSTMModel(input_size=300, hidden_size=64, output_size=1)
model.load_state_dict(torch.load("models/progression_model.pth"))
model.eval()

def predict_disease_progression(patient_data, image_path):
    """
    Predicts disease progression using LSTM and Bayesian probability.
    :param patient_data: List of health metrics.
    :param image_path: Path to the MRI/CT image file.
    :return: Predicted risk score & confidence level.
    """
    # Extract features from MRI/CT scan
    image_features = preprocess_image(image_path)

    # Combine patient health data with MRI/CT extracted features
    combined_input = np.concatenate((patient_data, image_features), axis=0)
    input_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0)

    # Get disease risk score from LSTM model
    risk_score = model(input_tensor).item()

    # Bayesian probability estimation
    confidence = norm.cdf(risk_score, loc=0.5, scale=0.1)

    # Save patient data to database
    save_patient_data(patient_data, image_features, risk_score, confidence)

    return {"risk_score": risk_score, "confidence": confidence}
