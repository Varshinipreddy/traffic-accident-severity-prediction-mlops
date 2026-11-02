import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = "data/processed"

COLLISION_FILE = os.path.join(
    PROCESSED_DIR,
    "collisions_cleaned.csv"
)

VEHICLE_FILE = os.path.join(
    PROCESSED_DIR,
    "vehicles_cleaned.csv"
)

CASUALTY_FILE = os.path.join(
    PROCESSED_DIR,
    "casualties_cleaned.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "accident_mart.csv"
)


# ============================================================
# BUILD ACCIDENT MART
# ============================================================

def build_accident_mart():

    print("=" * 60)
    print("BUILDING UPDATED ACCIDENT ANALYTICAL MART")
    print("=" * 60)

    # --------------------------------------------------------
    # Load collision data
    # --------------------------------------------------------

    print("\nLoading collision data...")

    collisions = pd.read_csv(
        COLLISION_FILE,
        low_memory=False
    )

    print(f"Collision rows: {len(collisions):,}")

    # --------------------------------------------------------
    # Load cleaned vehicle data
    # --------------------------------------------------------

    print("\nLoading cleaned vehicle data...")

    vehicles = pd.read_csv(
        VEHICLE_FILE,
        low_memory=False
    )

    print(f"Vehicle rows: {len(vehicles):,}")

    # --------------------------------------------------------
    # Load cleaned casualty data
    # --------------------------------------------------------

    print("\nLoading cleaned casualty data...")

    casualties = pd.read_csv(
        CASUALTY_FILE,
        low_memory=False
    )

    print(f"Casualty rows: {len(casualties):,}")

    # ========================================================
    # VEHICLE SUMMARY
    # ========================================================

    print("\nCreating vehicle summary...")

    vehicle_summary = (
        vehicles
        .groupby("collision_index")
        .agg(
            vehicle_count=(
                "vehicle_reference",
                "count"
            ),

            average_driver_age=(
                "driver_age_clean",
                "mean"
            ),

            female_driver_count=(
                "sex_of_driver",
                lambda x: (x == 2).sum()
            ),

            male_driver_count=(
                "sex_of_driver",
                lambda x: (x == 1).sum()
            ),

            escooter_vehicle_count=(
                "escooter_flag",
                lambda x: (x == 1).sum()
            )
        )
        .reset_index()
    )

    print(
        f"Vehicle summary rows: "
        f"{len(vehicle_summary):,}"
    )

    # ========================================================
    # CASUALTY SUMMARY
    # ========================================================

    print("\nCreating casualty summary...")

    casualty_summary = (
        casualties
        .groupby("collision_index")
        .agg(
            casualty_count=(
                "casualty_reference",
                "count"
            ),

            average_casualty_age=(
                "casualty_age_clean",
                "mean"
            ),

            fatal_casualties=(
                "casualty_severity",
                lambda x: (x == 1).sum()
            ),

            serious_casualties=(
                "casualty_severity",
                lambda x: (x == 2).sum()
            ),

            slight_casualties=(
                "casualty_severity",
                lambda x: (x == 3).sum()
            )
        )
        .reset_index()
    )

    print(
        f"Casualty summary rows: "
        f"{len(casualty_summary):,}"
    )

    # ========================================================
    # JOIN VEHICLE SUMMARY
    # ========================================================

    print("\nJoining vehicle summary...")

    mart = collisions.merge(
        vehicle_summary,
        on="collision_index",
        how="left",
        validate="one_to_one"
    )

    # ========================================================
    # JOIN CASUALTY SUMMARY
    # ========================================================

    print("Joining casualty summary...")

    mart = mart.merge(
        casualty_summary,
        on="collision_index",
        how="left",
        validate="one_to_one"
    )

    # ========================================================
    # FILL COUNT FIELDS
    # ========================================================

    count_columns = [
        "vehicle_count",
        "female_driver_count",
        "male_driver_count",
        "escooter_vehicle_count",
        "casualty_count",
        "fatal_casualties",
        "serious_casualties",
        "slight_casualties"
    ]

    for column in count_columns:

        if column in mart.columns:

            mart[column] = (
                mart[column]
                .fillna(0)
                .astype(int)
            )

    # ========================================================
    # ANALYTICAL FLAGS
    # ========================================================

    mart["has_fatal_casualty"] = (
        mart["fatal_casualties"] > 0
    ).astype(int)

    mart["has_serious_casualty"] = (
        mart["serious_casualties"] > 0
    ).astype(int)

    # ========================================================
    # CHECK UNIQUENESS
    # ========================================================

    duplicate_count = (
        mart["collision_index"]
        .duplicated()
        .sum()
    )

    print(
        f"\nDuplicate collision IDs: "
        f"{duplicate_count:,}"
    )

    if duplicate_count != 0:

        raise ValueError(
            "Analytical mart contains duplicate "
            "collision IDs."
        )

    # ========================================================
    # CHECK ROW COUNT
    # ========================================================

    if len(mart) != len(collisions):

        raise ValueError(
            "Mart row count does not match "
            "cleaned collision row count."
        )

    # ========================================================
    # SAVE
    # ========================================================

    mart.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("UPDATED ANALYTICAL MART CREATED")
    print("=" * 60)

    print(
        f"Rows: {len(mart):,}"
    )

    print(
        f"Columns: {len(mart.columns)}"
    )

    print(
        f"Duplicate collision IDs: "
        f"{mart['collision_index'].duplicated().sum()}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("\nAverage driver age:")
    print(
        mart["average_driver_age"].describe()
    )

    print("\nAverage casualty age:")
    print(
        mart["average_casualty_age"].describe()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_accident_mart()