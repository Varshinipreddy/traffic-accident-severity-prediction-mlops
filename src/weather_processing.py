import os
import glob
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

WEATHER_RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "weather",
    "raw"
)

WEATHER_STAGING_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "weather",
    "staging"
)

os.makedirs(WEATHER_STAGING_DIR, exist_ok=True)


def main():

    print("=" * 70)
    print("WEATHER PROCESSING")
    print("=" * 70)

    files = sorted(
        glob.glob(
            os.path.join(
                WEATHER_RAW_DIR,
                "open_meteo_daily_weather_*.csv"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            "No Open-Meteo daily weather files found."
        )

    print(f"\nWeather files found: {len(files)}")

    frames = []

    for file in files:

        print(
            f"Reading {os.path.basename(file)}..."
        )

        df = pd.read_csv(
            file,
            low_memory=False
        )

        frames.append(df)

        print(
            f"  Rows: {len(df):,}"
        )

    weather = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        f"\nCombined rows: {len(weather):,}"
    )

    # Standardize date
    weather["date"] = pd.to_datetime(
        weather["weather_date"],
        errors="coerce"
    ).dt.date
    # Standardize coordinates
    weather["weather_latitude"] = pd.to_numeric(
        weather["weather_latitude"],
        errors="coerce"
    ).round(2)

    weather["weather_longitude"] = pd.to_numeric(
        weather["weather_longitude"],
        errors="coerce"
    ).round(2)

    numeric_columns = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "weather_code",
        "wind_speed_10m_max",
    ]

    for col in numeric_columns:

        if col in weather.columns:

            weather[col] = pd.to_numeric(
                weather[col],
                errors="coerce"
            )

    # Remove invalid records
    before = len(weather)

    weather = weather.dropna(
        subset=[
            "date",
            "weather_latitude",
            "weather_longitude"
        ]
    )

    rejected = before - len(weather)

    print(
        f"Invalid weather records removed: {rejected:,}"
    )

    # Remove duplicates
    before = len(weather)

    weather = weather.drop_duplicates(
        subset=[
            "weather_latitude",
            "weather_longitude",
            "date"
        ]
    )

    duplicates = before - len(weather)

    print(
        f"Duplicate weather records removed: {duplicates:,}"
    )

    # Weather category
    def classify_weather(code):

        if pd.isna(code):
            return "Unknown"

        code = int(code)

        if code == 0:
            return "Clear"

        if code in [1, 2, 3]:
            return "Cloudy"

        if code in [45, 48]:
            return "Fog"

        if code in [51, 53, 55, 56, 57]:
            return "Drizzle"

        if code in [61, 63, 65, 66, 67]:
            return "Rain"

        if code in [71, 73, 75, 77, 85, 86]:
            return "Snow"

        if code in [80, 81, 82]:
            return "Rain Showers"

        if code in [95, 96, 99]:
            return "Thunderstorm"

        return "Other"

    weather["weather_label"] = weather[
        "weather_code"
    ].apply(classify_weather)

    # Source metadata
    weather["source"] = (
        "Open-Meteo Historical Weather API"
    )

    # Reorder columns
    columns = [
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
        "extraction_timestamp",
        "source",
    ]

    columns = [
        col for col in columns
        if col in weather.columns
    ]

    weather = weather[columns]

    weather = weather.sort_values(
        [
            "date",
            "weather_latitude",
            "weather_longitude"
        ]
    )

    output = os.path.join(
        WEATHER_STAGING_DIR,
        "weather_daily_staging.csv"
    )

    weather.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved staging file:"
    )
    print(output)

    print(
        f"Final weather rows: {len(weather):,}"
    )

    print(
        f"Unique locations: "
        f"{weather[['weather_latitude', 'weather_longitude']].drop_duplicates().shape[0]}"
    )

    print(
        f"Date range: "
        f"{weather['date'].min()} → {weather['date'].max()}"
    )

    print("\nRows by year:")

    print(
        weather.groupby(
            pd.to_datetime(weather["date"]).dt.year
        ).size()
    )

    print("\nWeather labels:")

    print(
        weather["weather_label"].value_counts()
    )

    print("\n" + "=" * 70)
    print("WEATHER PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()