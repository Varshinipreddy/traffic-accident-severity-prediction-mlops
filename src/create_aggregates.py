import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/processed/accident_mart.csv"
OUTPUT_DIR = "data/processed/aggregates"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_mart():

    print("Loading accident mart...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"Rows loaded: {len(df):,}")

    return df


# ============================================================
# HOURLY SUMMARY
# ============================================================

def create_hourly_summary(df):

    print("\nCreating hourly accident summary...")

    summary = (
        df.groupby("hour")
        .agg(
            total_accidents=("collision_index", "count"),
            fatal_accidents=(
                "severity_label",
                lambda x: (x == "Fatal").sum()
            ),
            serious_accidents=(
                "severity_label",
                lambda x: (x == "Serious").sum()
            ),
            slight_accidents=(
                "severity_label",
                lambda x: (x == "Slight").sum()
            ),
            total_casualties=(
                "number_of_casualties",
                "sum"
            )
        )
        .reset_index()
        .sort_values("hour")
    )

    output = os.path.join(
        OUTPUT_DIR,
        "hourly_accident_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(f"Created: {output}")

    return summary


# ============================================================
# DAILY SUMMARY
# ============================================================

def create_daily_summary(df):

    print("\nCreating daily accident summary...")

    summary = (
        df.groupby("weekday")
        .agg(
            total_accidents=("collision_index", "count"),
            fatal_accidents=(
                "severity_label",
                lambda x: (x == "Fatal").sum()
            ),
            serious_accidents=(
                "severity_label",
                lambda x: (x == "Serious").sum()
            ),
            slight_accidents=(
                "severity_label",
                lambda x: (x == "Slight").sum()
            ),
            total_casualties=(
                "number_of_casualties",
                "sum"
            )
        )
        .reset_index()
    )

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    summary["weekday"] = pd.Categorical(
        summary["weekday"],
        categories=weekday_order,
        ordered=True
    )

    summary = summary.sort_values("weekday")

    output = os.path.join(
        OUTPUT_DIR,
        "daily_accident_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(f"Created: {output}")

    return summary


# ============================================================
# ROAD TYPE SUMMARY
# ============================================================

def create_road_summary(df):

    print("\nCreating road type summary...")

    summary = (
        df.groupby("road_type_label")
        .agg(
            total_accidents=("collision_index", "count"),
            fatal_accidents=(
                "severity_label",
                lambda x: (x == "Fatal").sum()
            ),
            serious_accidents=(
                "severity_label",
                lambda x: (x == "Serious").sum()
            ),
            slight_accidents=(
                "severity_label",
                lambda x: (x == "Slight").sum()
            ),
            total_casualties=(
                "number_of_casualties",
                "sum"
            )
        )
        .reset_index()
        .sort_values(
            "total_accidents",
            ascending=False
        )
    )

    output = os.path.join(
        OUTPUT_DIR,
        "road_type_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(f"Created: {output}")

    return summary


# ============================================================
# WEATHER SUMMARY
# ============================================================

def create_weather_summary(df):

    print("\nCreating weather accident summary...")

    summary = (
        df.groupby("weather_label")
        .agg(
            total_accidents=("collision_index", "count"),
            fatal_accidents=(
                "severity_label",
                lambda x: (x == "Fatal").sum()
            ),
            serious_accidents=(
                "severity_label",
                lambda x: (x == "Serious").sum()
            ),
            slight_accidents=(
                "severity_label",
                lambda x: (x == "Slight").sum()
            ),
            total_casualties=(
                "number_of_casualties",
                "sum"
            )
        )
        .reset_index()
        .sort_values(
            "total_accidents",
            ascending=False
        )
    )

    output = os.path.join(
        OUTPUT_DIR,
        "weather_accident_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(f"Created: {output}")

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CREATING ANALYTICAL AGGREGATES")
    print("=" * 60)

    df = load_mart()

    hourly = create_hourly_summary(df)
    daily = create_daily_summary(df)
    road = create_road_summary(df)
    weather = create_weather_summary(df)

    print("\n" + "=" * 60)
    print("AGGREGATE CREATION COMPLETED")
    print("=" * 60)

    print("\nSummary:")
    print(f"Hourly rows:  {len(hourly)}")
    print(f"Daily rows:   {len(daily)}")
    print(f"Road rows:    {len(road)}")
    print(f"Weather rows: {len(weather)}")