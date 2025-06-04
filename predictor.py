import joblib
import numpy as np
from parameterized_features import extract_features
from cost_config import ROLE_COSTS, estimate_cost
from sklearn.metrics import r2_score

model = joblib.load("resource_predictor.pkl")

def predict_resources(description: str):
    features = extract_features(description)
    X = np.array([[features[role] for role in ROLE_COSTS.keys()]])

    prediction = model.predict(X)[0]
    resource_counts = {role: int(round(pred)) for role, pred in zip(ROLE_COSTS.keys(), prediction)}

    # Confidence intervals using trees from each role model
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

    # Cost calculation
    total_cost = estimate_cost(resource_counts)

    # Pseudo R² score
    dummy_y_true = X
    dummy_y_pred = model.predict(X)
    r2 = r2_score(dummy_y_true, dummy_y_pred)

    return resource_counts, intervals, total_cost, r2
