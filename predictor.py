import joblib
import numpy as np
from parameterized_features import extract_features
from cost_config import ROLE_COSTS, estimate_cost

# Load the trained model and its R² score
model, r2 = joblib.load("resource_predictor.pkl")

def predict_resources(description: str):
    """
    Predict resource allocation, confidence intervals, total cost, and model accuracy
    based on a project description.

    Parameters:
        description (str): Project description provided by the user.

    Returns:
        tuple:
            - resource_counts (dict): Estimated number of each resource type.
            - intervals (dict): 10th–90th percentile confidence range for each role.
            - total_cost (float): Estimated total project cost.
            - r2 (float): R² score from training (accuracy metric).
    """
    # Extract features from the input description
    features = extract_features(description)
    X = np.array([[features[role] for role in ROLE_COSTS.keys()]])

    # Predict resource counts
    prediction = model.predict(X)[0]
    resource_counts = {role: int(round(pred)) for role, pred in zip(ROLE_COSTS.keys(), prediction)}

    # Compute confidence intervals using predictions from all decision trees
    intervals = {}
    for i, role in enumerate(ROLE_COSTS.keys()):
        role_model = model.estimators_[i]
        tree_preds = [tree.predict(X)[0] for tree in role_model.estimators_]
        lb = int(np.floor(np.percentile(tree_preds, 10)))
        ub = int(np.ceil(np.percentile(tree_preds, 90)))
        intervals[role] = (lb, ub)

    # Estimate total cost
    total_cost = estimate_cost(resource_counts)

    return resource_counts, intervals, total_cost, r2
