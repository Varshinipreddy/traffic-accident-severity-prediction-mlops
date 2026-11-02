"""
Airflow DAG specification for the Traffic Accident ETL pipeline.

NOTE:
This project is executed on Windows using run_pipeline.py because
Apache Airflow is not installed in the Windows virtual environment.

The DAG below documents the equivalent Airflow orchestration that
can be deployed in an Airflow/Linux environment.
"""

from datetime import datetime

# Airflow is an optional orchestration dependency.
# The executable Windows pipeline is: python run_pipeline.py

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


if AIRFLOW_AVAILABLE:

    PROJECT_DIR = r"C:\Users\varsh\Downloads\traffic_accident_mlops_project"
    PYTHON = rf"{PROJECT_DIR}\.venv\Scripts\python.exe"

    with DAG(
        dag_id="traffic_accident_pipeline",
        start_date=datetime(2026, 1, 1),
        schedule="@weekly",
        catchup=False,
        tags=["data-engineering", "traffic-accident", "mlops"],
    ) as dag:

        ingest_accidents = BashOperator(
            task_id="ingest_accident_data",
            bash_command=(
                rf'cd /d "{PROJECT_DIR}" && '
                rf'"{PYTHON}" src\ingestion.py'
            ),
        )

        clean_accidents = BashOperator(
            task_id="clean_and_validate",
            bash_command=(
                rf'cd /d "{PROJECT_DIR}" && '
                rf'"{PYTHON}" src\cleaning.py'
            ),
        )

        build_accident_mart = BashOperator(
            task_id="build_accident_mart",
            bash_command=(
                rf'cd /d "{PROJECT_DIR}" && '
                rf'"{PYTHON}" src\build_mart.py'
            ),
        )

        create_aggregates = BashOperator(
            task_id="create_aggregates",
            bash_command=(
                rf'cd /d "{PROJECT_DIR}" && '
                rf'"{PYTHON}" src\create_aggregates.py'
            ),
        )

        build_weather_mart = BashOperator(
            task_id="build_weather_mart",
            bash_command=(
                rf'cd /d "{PROJECT_DIR}" && '
                rf'"{PYTHON}" src\build_weather_mart.py'
            ),
        )

        load_postgres = BashOperator(
            task_id="load_postgres",
            bash_command=(
                rf'cd /d "{PROJECT_DIR}" && '
                rf'"{PYTHON}" src\load_postgres.py'
            ),
        )

        load_open_meteo = BashOperator(
            task_id="load_open_meteo_weather",
            bash_command=(
                rf'cd /d "{PROJECT_DIR}" && '
                rf'"{PYTHON}" src\load_open_meteo_postgres.py'
            ),
        )

        ingest_accidents >> clean_accidents
        clean_accidents >> build_accident_mart
        build_accident_mart >> create_aggregates
        build_accident_mart >> build_weather_mart
        create_aggregates >> load_postgres
        build_weather_mart >> load_postgres
        load_postgres >> load_open_meteo