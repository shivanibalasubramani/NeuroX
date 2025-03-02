import torch
import torch.nn as nn
import numpy as np
from database.db import save_treatment_plan

# Define Deep Q-Network (DQN) Model
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define treatment actions (drugs, therapy, lifestyle)
ACTIONS = [
    "Drug A - Low Dosage", "Drug A - High Dosage",
    "Cognitive Therapy - Level 1", "Cognitive Therapy - Level 2",
    "Lifestyle Change - Diet", "Lifestyle Change - Exercise"
]

# Load pre-trained DQN model
model = DQNModel(state_size=1, action_size=len(ACTIONS))
model.load_state_dict(torch.load("models/treatment_model.pth"))
model.eval()

def recommend_treatment(risk_score):
    """
    Uses DQN to recommend a personalized treatment based on disease risk score.
    :param risk_score: Predicted disease risk score.
    :return: Recommended treatment & Q-value (decision confidence).
    """
    # Convert risk score into model input tensor
    state = torch.tensor([risk_score], dtype=torch.float32).unsqueeze(0)

    # Get Q-values for all treatment options
    q_values = model(state).detach().numpy()
    action_index = np.argmax(q_values)

    # Select best treatment recommendation
    treatment_plan = ACTIONS[action_index]

    # Save treatment plan to database
    save_treatment_plan(risk_score, treatment_plan, q_values.tolist())

    return {"treatment": treatment_plan, "q_value": q_values.tolist()}
