import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from parameterized_features import extract_features

roles = ["devs", "designers", "ai_agents", "legal_devs", "ai_specialists"]

# Training data descriptions
descriptions = [
    "mobile ecommerce app with ai chatbot and login",
    "real-time analytics dashboard with api integration",
    "hospital system with ai model and compliance checks",
    "banking software with data review and model training",
    "landing page for startup",
    "insurance platform for document management",
    "chatbot interface for ecommerce platform",
    "real-time hospital analytics with api",
    "secure bank login with legal compliance",
    "medical app for model-based diagnosis",
    "fraud detection model for bank customers",
    "cloud-based reporting dashboard for finance company",
    "insurance chatbot for claim prediction",
    "nlp-powered assistant for customer support",
    "machine learning pipeline for secure login systems",
    "documentation generator with analytics"
]

# Expand training data artificially
expanded_training_data = []
for desc in descriptions * 10:
    features = extract_features(desc)
    expanded_training_data.append({
        "description": desc,
        **features
    })

# DataFrames
df = pd.DataFrame(expanded_training_data)
X_df = df[roles]
y_df = df[roles]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Model training
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.5f}")

# Save model
joblib.dump(model, "resource_predictor.pkl")
