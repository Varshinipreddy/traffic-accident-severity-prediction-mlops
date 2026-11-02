import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "pipeline_run.log"


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    print(line)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, script):
    log("=" * 70)
    log(f"STARTING: {name}")
    log("=" * 70)

    result = subprocess.run(
        [PYTHON, str(PROJECT_DIR / "src" / script)],
        cwd=PROJECT_DIR,
    )

    if result.returncode != 0:
        log(f"FAILED: {name}")
        raise SystemExit(result.returncode)

    log(f"COMPLETED: {name}")


def main():
    start_time = datetime.now()

    log("=" * 70)
    log("TRAFFIC ACCIDENT ETL PIPELINE")
    log(f"Started: {start_time}")
    log("=" * 70)

    run_step(
        "Accident Data Ingestion",
        "ingestion.py",
    )

    run_step(
        "Data Cleaning and Validation",
        "cleaning.py",
    )

    run_step(
        "Build Accident Analytical Mart",
        "build_mart.py",
    )

    run_step(
        "Create Analytical Aggregates",
        "create_aggregates.py",
    )

    run_step(
        "Build Accident + Open-Meteo Weather Mart",
        "build_weather_mart.py",
    )

    run_step(
        "Load Accident Data into PostgreSQL/PostGIS",
        "load_postgres.py",
    )

    run_step(
        "Load Open-Meteo Weather into PostgreSQL",
        "load_open_meteo_postgres.py",
    )

    end_time = datetime.now()

    log("=" * 70)
    log("PIPELINE COMPLETED SUCCESSFULLY")
    log(f"Started : {start_time}")
    log(f"Finished: {end_time}")
    log(f"Log file: {LOG_FILE}")
    log("=" * 70)


if __name__ == "__main__":
    main()
    