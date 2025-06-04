import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from parameterized_features import extract_features

# Expanded labeled data for better model learning
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
        "devs": 2, "designers": 1, "ai_agents": 1, "legal_devs": 0, "ai_specialists": 1
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
    },
    {
        "description": "Legal compliance tool with automated contract generation.",
        "devs": 2, "designers": 1, "ai_agents": 1, "legal_devs": 2, "ai_specialists": 1
    },
    {
        "description": "AI-powered HR dashboard with employee attrition prediction.",
        "devs": 2, "designers": 1, "ai_agents": 2, "legal_devs": 0, "ai_specialists": 1
    },
    {
        "description": "NLP model deployment for sentiment analysis of product reviews.",
        "devs": 2, "designers": 1, "ai_agents": 2, "legal_devs": 0, "ai_specialists": 2
    },
    {
        "description": "Insurance claim processing automation with ML model.",
        "devs": 2, "designers": 1, "ai_agents": 2, "legal_devs": 1, "ai_specialists": 1
    },
    {
        "description": "Cybersecurity monitoring dashboard for internal IT infrastructure.",
        "devs": 3, "designers": 1, "ai_agents": 1, "legal_devs": 1, "ai_specialists": 0
    },
    {
        "description": "SaaS platform for project time tracking and invoicing.",
        "devs": 2, "designers": 2, "ai_agents": 0, "legal_devs": 0, "ai_specialists": 0
    },
    {
        "description": "AI-driven resume screening software for recruitment firms.",
        "devs": 2, "designers": 1, "ai_agents": 2, "legal_devs": 0, "ai_specialists": 1
    },
    {
        "description": "Build cloud-based photo storage and sharing application.",
        "devs": 3, "designers": 2, "ai_agents": 0, "legal_devs": 0, "ai_specialists": 0
    },
    {
        "description": "Financial planning SaaS with predictive modeling and charts.",
        "devs": 2, "designers": 2, "ai_agents": 1, "legal_devs": 1, "ai_specialists": 1
    },
    {
        "description": "Secure messaging platform with end-to-end encryption.",
        "devs": 3, "designers": 1, "ai_agents": 0, "legal_devs": 1, "ai_specialists": 0
    },
    {
        "description": "AI-based personal finance advisor chatbot.",
        "devs": 2, "designers": 1, "ai_agents": 2, "legal_devs": 0, "ai_specialists": 1
    },
    {
        "description": "Compliance reporting tool for government regulations.",
        "devs": 2, "designers": 1, "ai_agents": 1, "legal_devs": 2, "ai_specialists": 1
    }
]

# Extract features from descriptions
X = [extract_features(item["description"]) for item in training_data]
y = [[item["devs"], item["designers"], item["ai_agents"], item["legal_devs"], item["ai_specialists"]]
     for item in training_data]

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y, columns=["devs", "designers", "ai_agents", "legal_devs", "ai_specialists"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Multi-output Random Forest training
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Optional evaluation (not shown on UI)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# Save model (no R² included)
joblib.dump(model, "resource_predictor.pkl")
