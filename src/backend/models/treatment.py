import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from database.db import save_treatment_plan, log_treatment_history
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define Deep Q-Network (DQN) Model for Treatment Recommendation
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load Pre-trained DQN model
model = DQN(input_size=300, hidden_size=128, output_size=5)  # Output: 5 different treatment recommendations
model.load_state_dict(torch.load("models/treatment_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define Action Space (Treatment Options)
treatment_options = ["Medication Adjustment", "Cognitive Therapy", "Lifestyle Modification", "Physical Therapy", "Nutritional Plan"]

def recommend_treatment(risk_score, patient_data, previous_treatments):
    """
    Uses a Deep Q-Network (DQN) to recommend personalized treatment.
    :param risk_score: Predicted risk score from progression.py
    :param patient_data: List of health parameters (BP, cholesterol, heart rate, etc.)
    :param previous_treatments: List of past treatments used
    :return: Best treatment plan
    """
    
    try:
        logging.info("Processing patient data for treatment recommendation...")

        # Normalize previous treatments (One-Hot Encoding)
        previous_treatments_vector = np.zeros(len(treatment_options))
        for treatment in previous_treatments:
            if treatment in treatment_options:
                previous_treatments_vector[treatment_options.index(treatment)] = 1
        
        # Combine all inputs
        input_vector = np.concatenate((patient_data, [risk_score], previous_treatments_vector), axis=0)
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

        logging.info("Running DQN for treatment optimization...")
        q_values = model(input_tensor).detach().numpy().flatten()

        # Choose the treatment with the highest Q-value
        best_treatment = treatment_options[np.argmax(q_values)]
        confidence = softmax(q_values)[np.argmax(q_values)]

        # Save treatment recommendation in database
        save_treatment_plan(patient_data, risk_score, best_treatment, confidence)

        # Log treatment history
        log_treatment_history(patient_data, risk_score, best_treatment, confidence)

        logging.info(f"Recommended Treatment: {best_treatment} (Confidence: {confidence:.2%})")
        return {"treatment": best_treatment, "confidence": confidence}

    except Exception as e:
        logging.error(f"Error in treatment recommendation: {e}")
        return {"error": str(e)}

def softmax(x):
    """
    Computes softmax probabilities for Q-values.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

