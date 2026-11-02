import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

STAGING_DIR = "data/staging"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


VEHICLE_FILE = os.path.join(
    STAGING_DIR,
    "vehicles_staging.csv"
)

CASUALTY_FILE = os.path.join(
    STAGING_DIR,
    "casualties_staging.csv"
)

VEHICLE_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "vehicles_cleaned.csv"
)

CASUALTY_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "casualties_cleaned.csv"
)


# ============================================================
# VEHICLE CLEANING
# ============================================================

def clean_vehicle_data():

    print("\n" + "=" * 60)
    print("CLEANING VEHICLE DATA")
    print("=" * 60)

    vehicles = pd.read_csv(
        VEHICLE_FILE,
        low_memory=False
    )

    print(f"Original vehicle rows: {len(vehicles):,}")

    # --------------------------------------------------------
    # Driver age
    # --------------------------------------------------------
    #
    # STATS19 uses special values for unavailable/not known
    # ages. Keep valid ages only for numerical calculations.
    #

    vehicles["driver_age_clean"] = pd.to_numeric(
        vehicles["age_of_driver"],
        errors="coerce"
    )

    vehicles.loc[
        ~vehicles["driver_age_clean"].between(10, 100),
        "driver_age_clean"
    ] = pd.NA

    # --------------------------------------------------------
    # Driver sex
    # --------------------------------------------------------

    vehicles["driver_sex_label"] = vehicles[
        "sex_of_driver"
    ].map({
        1: "Male",
        2: "Female"
    })

    # --------------------------------------------------------
    # Vehicle type
    # --------------------------------------------------------

    vehicles["vehicle_type_label"] = vehicles[
        "vehicle_type"
    ].map({
        1: "Pedal cycle",
        2: "Motorcycle 50cc and under",
        3: "Motorcycle over 50cc and up to 125cc",
        4: "Motorcycle over 125cc and up to 500cc",
        5: "Motorcycle over 500cc",
        8: "Taxi/Private hire car",
        9: "Car",
        10: "Minibus",
        11: "Bus/Coach",
        16: "Horse rider",
        17: "Agricultural vehicle",
        18: "Tram",
        19: "Van/Goods vehicle",
        20: "Goods vehicle over 3.5 tonnes",
        21: "Mobility scooter",
        22: "Electric motorcycle",
        23: "Electric scooter"
    })

    # --------------------------------------------------------
    # Propulsion
    # --------------------------------------------------------

    vehicles["propulsion_label"] = vehicles[
        "propulsion_code"
    ].map({
        1: "Petrol",
        2: "Heavy oil",
        3: "Electric",
        4: "Steam",
        5: "Gas",
        6: "Petrol/Gas",
        7: "Gas/Bi-fuel",
        8: "Electric hybrid"
    })

    # --------------------------------------------------------
    # Vehicle age
    # --------------------------------------------------------

    vehicles["vehicle_age_clean"] = pd.to_numeric(
        vehicles["age_of_vehicle"],
        errors="coerce"
    )

    vehicles.loc[
        ~vehicles["vehicle_age_clean"].between(0, 100),
        "vehicle_age_clean"
    ] = pd.NA

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    vehicles.to_csv(
        VEHICLE_OUTPUT,
        index=False
    )

    print(f"Cleaned vehicle rows: {len(vehicles):,}")
    print(f"Output: {VEHICLE_OUTPUT}")

    return vehicles


# ============================================================
# CASUALTY CLEANING
# ============================================================

def clean_casualty_data():

    print("\n" + "=" * 60)
    print("CLEANING CASUALTY DATA")
    print("=" * 60)

    casualties = pd.read_csv(
        CASUALTY_FILE,
        low_memory=False
    )

    print(
        f"Original casualty rows: "
        f"{len(casualties):,}"
    )

    # --------------------------------------------------------
    # Casualty age
    # --------------------------------------------------------

    casualties["casualty_age_clean"] = pd.to_numeric(
        casualties["age_of_casualty"],
        errors="coerce"
    )

    casualties.loc[
        ~casualties["casualty_age_clean"].between(0, 100),
        "casualty_age_clean"
    ] = pd.NA

    # --------------------------------------------------------
    # Casualty sex
    # --------------------------------------------------------

    casualties["casualty_sex_label"] = casualties[
        "sex_of_casualty"
    ].map({
        1: "Male",
        2: "Female"
    })

    # --------------------------------------------------------
    # Casualty class
    # --------------------------------------------------------

    casualties["casualty_class_label"] = casualties[
        "casualty_class"
    ].map({
        1: "Driver or rider",
        2: "Passenger",
        3: "Pedestrian"
    })

    # --------------------------------------------------------
    # Casualty severity
    # --------------------------------------------------------

    casualties["casualty_severity_label"] = casualties[
        "casualty_severity"
    ].map({
        1: "Fatal",
        2: "Serious",
        3: "Slight"
    })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    casualties.to_csv(
        CASUALTY_OUTPUT,
        index=False
    )

    print(
        f"Cleaned casualty rows: "
        f"{len(casualties):,}"
    )

    print(
        f"Output: {CASUALTY_OUTPUT}"
    )

    return casualties


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    clean_vehicle_data()
    clean_casualty_data()

    print("\n" + "=" * 60)
    print("VEHICLE + CASUALTY CLEANING COMPLETED")
    print("=" * 60)