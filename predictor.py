import joblib
import numpy as np
from parameterized_features import extract_features
from cost_config import ROLE_COSTS, estimate_cost

# Load trained model (no R² score needed anymore)
model = joblib.load("resource_predictor.pkl")

def predict_resources(description: str):
    # Extract features from the input description
    features = extract_features(description)
    X = np.array([[features[role] for role in ROLE_COSTS.keys()]])

    # Predict the required number of each resource
    prediction = model.predict(X)[0]
    resource_counts = {
        role: int(round(pred)) for role, pred in zip(ROLE_COSTS.keys(), prediction)
    }

    # Estimate confidence range (10th to 90th percentile) using tree predictions
    intervals = {}
    for i, role in enumerate(ROLE_COSTS.keys()):
        role_model = model.estimators_[i]
        tree_preds = [
            tree.predict(X)[0]
            for tree in role_model.estimators_
        ]
        lb = int(np.floor(np.percentile(tree_preds, 10)))
        ub = int(np.ceil(np.percentile(tree_preds, 90)))
        intervals[role] = (lb, ub)

    # Calculate total estimated cost
    total_cost = estimate_cost(resource_counts)

    return resource_counts, intervals, total_cost
