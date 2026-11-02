# Traffic Accident Severity Analytics and Prediction

A student-friendly MLOps/Data Engineering project based on the assignment brief.

## Part 1
- Python/Pandas ingestion
- Raw, staging, cleaned and analytical layers
- Data quality checks and rejected-record log
- PostgreSQL loading
- Streamlit dashboard
- Airflow DAG

## Quick start
1. Create a Python virtual environment.
2. Install `requirements.txt`.
3. Put your accident CSV in `data/raw/accidents.csv`.
4. Run:
   `python src/ingestion.py`
   `python src/cleaning.py`
   `python src/build_mart.py`
5. Run dashboard:
   `streamlit run dashboard/app.py`

The pipeline includes a demo-data fallback if `data/raw/accidents.csv` is not present, so you can test the project immediately. Replace the demo data with your approved public dataset before submission.

## PostgreSQL
Set these environment variables before loading:
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Then run:
`python src/load_postgres.py`

## Airflow
Copy/use `dags/accident_pipeline.py` in your Airflow DAGs folder.
