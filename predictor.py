import joblib
import numpy as np
from parameterized_features import extract_features
from cost_config import estimate_cost

# Load model
model = joblib.load("resource_predictor.pkl")

def predict_resources(description: str):
    features = extract_features(description)
    X = [list(features.values())]

    # Base prediction
    raw_pred = model.predict(X)[0]
    resource_counts = {
        "devs": round(raw_pred[0]),
        "designers": round(raw_pred[1]),
        "ai_agents": round(raw_pred[2]),
        "legal_devs": round(raw_pred[3]),
        "ai_specialists": round(raw_pred[4])
    }

    # Confidence intervals (10th to 90th percentile)
    lower_bounds = {}
    upper_bounds = {}
    for i, role in enumerate(resource_counts.keys()):
        role_model = model.estimators_[i]
        tree_preds = [tree.predict(X)[0] for tree in role_model.estimators_]
        lower_bounds[role] = max(0, round(np.percentile(tree_preds, 10)))
        upper_bounds[role] = round(np.percentile(tree_preds, 90))

    intervals = {
        role: (lower_bounds[role], upper_bounds[role])
        for role in resource_counts.keys()
    }

    total_cost = estimate_cost(resource_counts)
    return resource_counts, intervals, total_cost