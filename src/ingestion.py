import os
from datetime import datetime

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

RAW_DIR = "data/raw"
STAGING_DIR = "data/staging"

os.makedirs(STAGING_DIR, exist_ok=True)


# ============================================================
# REAL DfT DATASET FILES
# ============================================================

COLLISION_FILE = os.path.join(
    RAW_DIR,
    "dft-road-casualty-statistics-collision-last-5-years.csv"
)

VEHICLE_FILE = os.path.join(
    RAW_DIR,
    "dft-road-casualty-statistics-vehicle-last-5-years.csv"
)

CASUALTY_FILE = os.path.join(
    RAW_DIR,
    "dft-road-casualty-statistics-casualty-last-5-years.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

COLLISION_OUTPUT = os.path.join(
    STAGING_DIR,
    "collisions_staging.csv"
)

VEHICLE_OUTPUT = os.path.join(
    STAGING_DIR,
    "vehicles_staging.csv"
)

CASUALTY_OUTPUT = os.path.join(
    STAGING_DIR,
    "casualties_staging.csv"
)

LOG_OUTPUT = os.path.join(
    STAGING_DIR,
    "ingestion_log.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def check_file(file_path):
    """Check whether a source file exists."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )


# ============================================================
# INGESTION FUNCTION
# ============================================================

def ingest_data():

    print("=" * 60)
    print("TRAFFIC ACCIDENT DATA INGESTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check source files
    # --------------------------------------------------------

    print("\nChecking source files...")

    check_file(COLLISION_FILE)
    check_file(VEHICLE_FILE)
    check_file(CASUALTY_FILE)

    print("✓ All three source files found.")

    # --------------------------------------------------------
    # Read collision data
    # --------------------------------------------------------

    print("\nReading collision dataset...")

    collisions = pd.read_csv(
        COLLISION_FILE,
        low_memory=False
    )

    print(
        f"✓ Collision records loaded: "
        f"{len(collisions):,}"
    )

    # --------------------------------------------------------
    # Read vehicle data
    # --------------------------------------------------------

    print("\nReading vehicle dataset...")

    vehicles = pd.read_csv(
        VEHICLE_FILE,
        low_memory=False
    )

    print(
        f"✓ Vehicle records loaded: "
        f"{len(vehicles):,}"
    )

    # --------------------------------------------------------
    # Read casualty data
    # --------------------------------------------------------

    print("\nReading casualty dataset...")

    casualties = pd.read_csv(
        CASUALTY_FILE,
        low_memory=False
    )

    print(
        f"✓ Casualty records loaded: "
        f"{len(casualties):,}"
    )

    # --------------------------------------------------------
    # Save staging copies
    # --------------------------------------------------------

    print("\nCreating staging layer...")

    collisions.to_csv(
        COLLISION_OUTPUT,
        index=False
    )

    vehicles.to_csv(
        VEHICLE_OUTPUT,
        index=False
    )

    casualties.to_csv(
        CASUALTY_OUTPUT,
        index=False
    )

    print("✓ Collision staging file created.")
    print("✓ Vehicle staging file created.")
    print("✓ Casualty staging file created.")

    # --------------------------------------------------------
    # Create ingestion log
    # --------------------------------------------------------

    ingestion_log = pd.DataFrame([
        {
            "extraction_timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "source_file": os.path.basename(COLLISION_FILE),
            "dataset_type": "collision",
            "status": "SUCCESS",
            "row_count": len(collisions)
        },
        {
            "extraction_timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "source_file": os.path.basename(VEHICLE_FILE),
            "dataset_type": "vehicle",
            "status": "SUCCESS",
            "row_count": len(vehicles)
        },
        {
            "extraction_timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "source_file": os.path.basename(CASUALTY_FILE),
            "dataset_type": "casualty",
            "status": "SUCCESS",
            "row_count": len(casualties)
        }
    ])

    ingestion_log.to_csv(
        LOG_OUTPUT,
        index=False
    )

    print("\n✓ Ingestion log created.")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ingest_data()