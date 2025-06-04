import joblib
import numpy as np
from parameterized_features import extract_features
from cost_config import ROLE_COSTS, estimate_cost

model = joblib.load("resource_predictor.pkl")

def predict_resources(description: str):
    # Extract raw features from input
    features = extract_features(description)

    # Prepare input for model
    X = np.array([[features[role] for role in ROLE_COSTS.keys()]])
    prediction = model.predict(X)[0]

    # Force prediction = 0 for roles with zero feature score
    resource_counts = {
        role: int(round(pred)) if features[role] > 0 else 0
        for role, pred in zip(ROLE_COSTS.keys(), prediction)
    }

    # Confidence intervals (10–90th percentile from decision trees)
    intervals = {}
    for i, role in enumerate(ROLE_COSTS.keys()):
        role_model = model.estimators_[i]
        tree_preds = [tree.predict(X)[0] for tree in role_model.estimators_]
        lb = int(np.floor(np.percentile(tree_preds, 10))) if features[role] > 0 else 0
        ub = int(np.ceil(np.percentile(tree_preds, 90))) if features[role] > 0 else 0
        intervals[role] = (lb, ub)

    # Calculate estimated cost
    total_cost = estimate_cost(resource_counts)

    return resource_counts, intervals, total_cost
