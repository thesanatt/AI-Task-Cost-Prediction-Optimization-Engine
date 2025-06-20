AI Task Cost Prediction & Optimization Engine
============================================

Overview
--------
This project predicts the types and counts of human/AI resources required to complete a software project, based on a natural-language task description. It also estimates the total cost using role-wise rates and a 4-week project duration.

Hosted on: https://ai-task-cost-prediction-engine.streamlit.app/

It uses:
- A machine learning model (Random Forest) trained with keyword-based features
- Cost estimation using configurable role-wise costs and project duration
- Confidence intervals per role using ensemble tree predictions
- A Streamlit web dashboard for input and output

How to Run Locally
------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train.py

3. Launch the dashboard:
   streamlit run main.py --server.port 8502

Key Concepts Used
-----------------
- Random Forest Regression
- MultiOutput Regression
- Confidence ranges from decision trees
- Feature extraction using keyword scores
- Cost modeling via custom rates
- Streamlit dashboard

## Cost Assumptions

- Developers: $2000/week  
- Designers: $1500/week  
- AI Agents: $2500/week  
- Legal Devs: $2200/week  
- AI Specialists: $3000/week  
- **Project Duration**: 4 weeks (constant)

Built With
----------
- Python 3.10+
- scikit-learn
- pandas
- numpy
- streamlit
- joblib
