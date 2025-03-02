import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.stats import norm
from utils.image_processing import preprocess_image, enhance_contrast, detect_edges
from database.db import save_patient_data, log_progression_history
from datetime import datetime
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define LSTM Model with Dropout and Batch Normalization for better generalization
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.batch_norm(lstm_out[:, -1, :])
        return self.fc(lstm_out)

# Load pre-trained LSTM model
model = LSTMModel(input_size=350, hidden_size=128, output_size=1)  # Increased input size for extra features
model.load_state_dict(torch.load("models/progression_model.pth", map_location=torch.device("cpu")))
model.eval()

def predict_disease_progression(patient_data, image_path, lab_report=None, previous_records=None):
    """
    Predicts disease progression using LSTM and Bayesian probability.
    :param patient_data: List of numerical health metrics (heart rate, BP, cholesterol, etc.).
    :param image_path: Path to the MRI/CT image file.
    :param lab_report: Optional list of lab test results.
    :param previous_records: Optional list of past progression scores.
    :return: Predicted risk score & confidence level.
    """
    
    try:
        logging.info("Processing MRI/CT image for feature extraction...")
        image_features = preprocess_image(image_path)

        logging.info("Enhancing image contrast and detecting critical features...")
        enhanced_image = enhance_contrast(image_path)
        edge_features = detect_edges(image_path)

        # Normalize additional data
        lab_report = np.array(lab_report) if lab_report else np.zeros(10)  # 10 lab features (default 0)
        previous_records = np.array(previous_records) if previous_records else np.zeros(5)  # Last 5 records

        # Combine all inputs
        combined_input = np.concatenate((patient_data, image_features, edge_features, lab_report, previous_records), axis=0)
        input_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0)

        logging.info("Running LSTM model for disease progression prediction...")
        risk_score = model(input_tensor).item()

        # Advanced Bayesian probability estimation
        confidence = dynamic_bayesian_probability(risk_score, previous_records)

        # Save patient data to database
        save_patient_data(patient_data, image_features, edge_features, risk_score, confidence, lab_report)

        # Log progression history
        log_progression_history(patient_data, risk_score, confidence)

        logging.info(f"Prediction Complete - Risk Score: {risk_score:.4f}, Confidence: {confidence:.2%}")
        return {"risk_score": risk_score, "confidence": confidence, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logging.error(f"Error in disease progression prediction: {e}")
        return {"error": str(e)}

def dynamic_bayesian_probability(risk_score, history):
    """
    Custom Bayesian probability estimation considering historical trends.
    :param risk_score: Current risk score.
    :param history: Past disease progression records.
    :return: Adjusted confidence estimation.
    """
    baseline = np.mean(history) if len(history) > 0 else 0.5
    deviation = np.std(history) if len(history) > 0 else 0.1
    return norm.cdf(risk_score, loc=baseline, scale=deviation)

