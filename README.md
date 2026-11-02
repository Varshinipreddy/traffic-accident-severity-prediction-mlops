# traffic-accident-severity-prediction-mlops

# Traffic Accident Severity Analytics and Prediction

This project analyzes road traffic accidents and predicts accident severity using machine learning.

## Project Overview

The project uses UK road accident data and historical weather data to analyze accident patterns and predict collision severity.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- XGBoost
- LightGBM
- PostgreSQL
- Streamlit
- FastAPI
- Apache Airflow
- MLflow

## Features

- Accident data analysis and visualization
- Accident severity prediction
- Weather data integration
- Interactive Streamlit dashboard
- Machine learning model comparison
- MLflow experiment tracking
- Automated ML pipeline using Airflow
- FastAPI prediction service
- API performance monitoring

## Machine Learning Models

- Random Forest
- XGBoost
- LightGBM

Random Forest was selected based on its macro F1-score.

## Project Structure

```text
traffic-accident-severity-prediction/
├── api/
├── dashboard/
├── ml/
├── dags/
├── models/
├── monitoring/
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
python -m streamlit run dashboard/app.py
```

Run the FastAPI application:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

## Author

**Varshini Papa Reddy**

B.Tech – Artificial Intelligence and Machine Learning

CMR Institute of Technology, Bengaluru
