AI Task Cost Prediction & Optimization Engine
============================================

Overview
--------
This project predicts the number of team members (developers, designers, AI agents, etc.) and the cost required to build an AI-related software project, based on a project description.

It uses:
- A machine learning model (Random Forest) trained with keyword-based features
- Cost estimation using configurable role-wise costs and project duration
- Confidence intervals per role using ensemble tree predictions
- A Streamlit web dashboard for input and output
- Model accuracy via R² score

Example
-------
Input:
"Create a secure fintech platform with AI fraud detection and compliance features."

Output:
- Developers: 3 (range: 2–4)
- Designers: 1 (range: 0–1)
- AI Agents: 2 (range: 1–3)
- Legal Devs: 1 (range: 1–2)
- AI Specialists: 2 (range: 1–3)
- Estimated Total Cost: $54,000
- R² Score: 0.92

How to Run Locally
------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train.py

3. Launch the dashboard:
   streamlit run main.py --server.port 8502

Accuracy Metrics
----------------
- R² Score: Evaluates how well predictions match actual values.

- Confidence Intervals: Computed using tree ensemble percentiles (10th–90th).

Key Concepts Used
-----------------
- Random Forest Regression
- MultiOutput Regression
- R² score
- Confidence ranges from decision trees
- Feature extraction using keyword scores
- Cost modeling via custom rates
- Streamlit dashboard

Built With
----------
- Python 3.10+
- scikit-learn
- pandas
- numpy
- streamlit
- joblib
