import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from parameterized_features import extract_features

# Sample labeled data for training
training_data = [
    {
        "description": "Build a mobile banking app with real-time AI chatbot integration and high security.",
        "devs": 3, "designers": 1, "ai_agents": 3, "legal_devs": 1, "ai_specialists": 1
    },
    {
        "description": "Create a landing page for a small business.",
        "devs": 1, "designers": 1, "ai_agents": 0, "legal_devs": 0, "ai_specialists": 0
    },
    {
        "description": "Develop an AI-based legal document review tool for a finance firm.",
        "devs": 2, "designers": 1, "ai_agents": 2, "legal_devs": 1, "ai_specialists": 1
    },
    {
        "description": "Interactive dashboard for hospital analytics system.",
        "devs": 2, "designers": 1, "ai_agents": 1, "legal_devs": 0, "ai_specialists": 0
    },
    {
        "description": "Urgent mobile ecommerce platform with login and API integrations.",
        "devs": 3, "designers": 2, "ai_agents": 1, "legal_devs": 0, "ai_specialists": 0
    },
    {
        "description": "Banking API with fraud detection using AI and legal compliance requirements.",
        "devs": 3, "designers": 1, "ai_agents": 2, "legal_devs": 1, "ai_specialists": 1
    },
    {
        "description": "Healthcare chatbot with medical database integration.",
        "devs": 2, "designers": 1, "ai_agents": 2, "legal_devs": 0, "ai_specialists": 1
    },
    {
        "description": "Simple blog platform for a travel writer.",
        "devs": 1, "designers": 1, "ai_agents": 0, "legal_devs": 0, "ai_specialists": 0
    }
]

# Feature extraction and target preparation
X = [extract_features(item["description"]) for item in training_data]
y = [[item["devs"], item["designers"], item["ai_agents"], item["legal_devs"], item["ai_specialists"]]
     for item in training_data]

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y, columns=["devs", "designers", "ai_agents", "legal_devs", "ai_specialists"])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Train a MultiOutput Random Forest model
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Evaluate model performance
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model and R² score together for reuse
joblib.dump((model, r2), "resource_predictor.pkl")
