import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ACCIDENT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "accident_mart.csv"
)

WEATHER_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "weather",
    "staging",
    "weather_daily_staging.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "accident_weather_mart.csv"
)

GRID_SIZE = 0.5


def create_weather_grid(df):
    df["weather_latitude"] = (
        np.round(
            df["latitude"] / GRID_SIZE
        ) * GRID_SIZE
    ).round(2)

    df["weather_longitude"] = (
        np.round(
            df["longitude"] / GRID_SIZE
        ) * GRID_SIZE
    ).round(2)

    return df


def main():

    print("=" * 70)
    print("BUILD ACCIDENT + WEATHER ANALYTICAL MART")
    print("=" * 70)

    print("\nReading accident mart...")

    accidents = pd.read_csv(
        ACCIDENT_FILE,
        low_memory=False
    )

    print(
        f"Accident rows: {len(accidents):,}"
    )

    print("\nReading Open-Meteo weather...")

    weather = pd.read_csv(
        WEATHER_FILE,
        low_memory=False
    )

    print(
        f"Weather rows: {len(weather):,}"
    )

    # ---------------------------------------------------------
    # Standardize dates
    # ---------------------------------------------------------

    accidents["date"] = pd.to_datetime(
        accidents["date"],
        errors="coerce"
    ).dt.date

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce"
    ).dt.date

    # ---------------------------------------------------------
    # Create same 0.5 degree grid used by weather ingestion
    # ---------------------------------------------------------

    accidents = create_weather_grid(
        accidents
    )

    weather["weather_latitude"] = pd.to_numeric(
        weather["weather_latitude"],
        errors="coerce"
    ).round(2)

    weather["weather_longitude"] = pd.to_numeric(
        weather["weather_longitude"],
        errors="coerce"
    ).round(2)

    # ---------------------------------------------------------
    # Select weather columns
    # ---------------------------------------------------------

    weather_columns = [
        "date",
        "weather_latitude",
        "weather_longitude",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "weather_code",
        "weather_label",
        "wind_speed_10m_max",
    ]

    weather = weather[
        [
            c
            for c in weather_columns
            if c in weather.columns
        ]
    ]

    # ---------------------------------------------------------
    # Remove accidental duplicate weather keys
    # ---------------------------------------------------------

    weather = weather.drop_duplicates(
        subset=[
            "date",
            "weather_latitude",
            "weather_longitude",
        ]
    )

    print(
        f"Weather rows after deduplication: "
        f"{len(weather):,}"
    )

    # ---------------------------------------------------------
    # Accident + weather JOIN
    #
    # TIME  = accident date
    # LOCATION = 0.5 degree weather grid
    # ---------------------------------------------------------

    print("\nJoining accident and weather data...")

    result = accidents.merge(
        weather,
        how="left",
        on=[
            "date",
            "weather_latitude",
            "weather_longitude",
        ],
        indicator=True
    )

    # ---------------------------------------------------------
    # Match status
    # ---------------------------------------------------------

    result["weather_match_status"] = np.where(
        result["_merge"] == "both",
        "Matched",
        "Weather unavailable"
    )

    result = result.drop(
        columns=["_merge"]
    )
    result = result.rename(
        columns={
            "weather_label_x": "dft_weather_label",
            "weather_label_y": "open_meteo_weather_label"
        }
    )

    # ---------------------------------------------------------
    # Weather coverage summary
    # ---------------------------------------------------------

    total = len(result)

    matched = (
        result["weather_match_status"]
        == "Matched"
    ).sum()

    unmatched = (
        result["weather_match_status"]
        == "Weather unavailable"
    ).sum()

    print("\n" + "=" * 70)
    print("WEATHER JOIN RESULTS")
    print("=" * 70)

    print(
        f"Total accident records: {total:,}"
    )

    print(
        f"Weather matched: {matched:,}"
    )

    print(
        f"Weather unavailable: {unmatched:,}"
    )

    print(
        f"Match rate: {matched / total * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Match rate by year
    # ---------------------------------------------------------

    result["year"] = pd.to_datetime(
        result["date"],
        errors="coerce"
    ).dt.year

    yearly = (
        result.groupby(
            "year",
            dropna=False
        )
        .agg(
            accident_records=(
                "collision_index",
                "count"
            ),
            weather_matched=(
                "weather_match_status",
                lambda x: (
                    x == "Matched"
                ).sum()
            )
        )
        .reset_index()
    )

    yearly["match_rate_percent"] = (
        yearly["weather_matched"]
        / yearly["accident_records"]
        * 100
    )

    print("\nWeather coverage by year:")
    print(
        yearly.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    print(
        f"Final rows: {len(result):,}"
    )

    print(
        f"Final columns: {len(result.columns)}"
    )

    print("\n" + "=" * 70)
    print("WEATHER MART COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()