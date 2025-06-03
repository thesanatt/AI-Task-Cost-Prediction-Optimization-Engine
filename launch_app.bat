@echo off
echo Checking/installing required libraries...
pip install -r requirements.txt

echo Starting Streamlit App...
streamlit run main.py --server.port=8502
pause