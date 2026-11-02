import os
import time
import requests
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

WEATHER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "weather",
    "raw"
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

GRID_SIZE = 0.5

# Small batches reduce the chance of API rate limiting.
BATCH_SIZE = 5

# Delay between successful requests.
DELAY_SECONDS = 10

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "weather_code",
    "wind_speed_10m_max",
]


def weather_grid(df):
    df = df[["latitude", "longitude"]].dropna().copy()

    df["weather_latitude"] = (
        np.round(df["latitude"] / GRID_SIZE) * GRID_SIZE
    ).round(2)

    df["weather_longitude"] = (
        np.round(df["longitude"] / GRID_SIZE) * GRID_SIZE
    ).round(2)

    return (
        df[
            ["weather_latitude", "weather_longitude"]
        ]
        .drop_duplicates()
        .sort_values(
            ["weather_latitude", "weather_longitude"]
        )
        .reset_index(drop=True)
    )


def load_existing(year):
    path = os.path.join(
        WEATHER_DIR,
        f"open_meteo_daily_weather_{year}.csv"
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        low_memory=False
    )

    if df.empty:
        return pd.DataFrame()

    df["weather_latitude"] = pd.to_numeric(
        df["weather_latitude"],
        errors="coerce"
    ).round(2)

    df["weather_longitude"] = pd.to_numeric(
        df["weather_longitude"],
        errors="coerce"
    ).round(2)

    if "weather_date" in df.columns:
        df["weather_date"] = pd.to_datetime(
            df["weather_date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    return df


def fetch_batch(locations, year):
    lats = ",".join(
        str(x[0]) for x in locations
    )

    lons = ",".join(
        str(x[1]) for x in locations
    )

    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/London",
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            timeout=120
        )

        if response.status_code == 429:
            print("  API rate limit (429).")
            print("  Waiting 120 seconds before continuing...")
            time.sleep(120)
            return pd.DataFrame()

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            data = [data]

        parts = []

        for item in data:

            if "daily" not in item:
                continue

            daily = item["daily"]

            part = pd.DataFrame(daily)

            if part.empty:
                continue

            if "time" not in part.columns:
                continue

            part = part.rename(
                columns={
                    "time": "weather_date"
                }
            )

            part["weather_latitude"] = round(
                float(item["latitude"]),
                2
            )

            part["weather_longitude"] = round(
                float(item["longitude"]),
                2
            )

            parts.append(part)

        if not parts:
            return pd.DataFrame()

        result = pd.concat(
            parts,
            ignore_index=True
        )

        result["weather_date"] = pd.to_datetime(
            result["weather_date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        return result

    except Exception as e:
        print(f"  Request failed: {e}")
        return pd.DataFrame()


def save_year(year, df):
    output = os.path.join(
        WEATHER_DIR,
        f"open_meteo_daily_weather_{year}.csv"
    )

    df["weather_latitude"] = pd.to_numeric(
        df["weather_latitude"],
        errors="coerce"
    ).round(2)

    df["weather_longitude"] = pd.to_numeric(
        df["weather_longitude"],
        errors="coerce"
    ).round(2)

    df = df.drop_duplicates(
        subset=[
            "weather_latitude",
            "weather_longitude",
            "weather_date"
        ]
    )

    df["extraction_timestamp"] = pd.Timestamp.now(
        tz="UTC"
    )

    df["source"] = (
        "Open-Meteo Historical Weather API"
    )

    df.to_csv(
        output,
        index=False
    )

    return output, df


def main():

    print("=" * 70)
    print("OPEN-METEO WEATHER REPAIR")
    print("2024-2025 ONLY")
    print("=" * 70)

    os.makedirs(
        WEATHER_DIR,
        exist_ok=True
    )

    print("\nLoading accident locations...")

    accidents = pd.read_csv(
        ACCIDENT_FILE,
        usecols=[
            "latitude",
            "longitude",
            "date"
        ],
        low_memory=False
    )

    accidents["date"] = pd.to_datetime(
        accidents["date"],
        errors="coerce"
    )

    accidents = accidents.dropna(
        subset=[
            "latitude",
            "longitude",
            "date"
        ]
    )

    grid = weather_grid(accidents)

    required = set(
        zip(
            grid["weather_latitude"],
            grid["weather_longitude"]
        )
    )

    print(
        f"Required weather locations: {len(required)}"
    )

    for year in [2024, 2025]:

        print("\n" + "=" * 70)
        print(f"YEAR {year}")
        print("=" * 70)

        existing = load_existing(year)

        if existing.empty:
            existing_locations = set()
        else:
            existing_locations = set(
                zip(
                    existing["weather_latitude"],
                    existing["weather_longitude"]
                )
            )

        missing = sorted(
            required - existing_locations
        )

        print(
            f"Existing locations: {len(existing_locations)}"
        )

        print(
            f"Missing locations: {len(missing)}"
        )

        if not missing:
            print("Year already complete.")
            continue

        print(
            f"Downloading {len(missing)} locations..."
        )

        for start in range(
            0,
            len(missing),
            BATCH_SIZE
        ):

            batch = missing[
                start:start + BATCH_SIZE
            ]

            end = min(
                start + BATCH_SIZE,
                len(missing)
            )

            print(
                f"\nRequest {start + 1}-{end} "
                f"of {len(missing)}"
            )

            result = fetch_batch(
                batch,
                year
            )

            if result.empty:

                print(
                    "  No data received."
                )

                print(
                    "  Skipping this batch."
                )

                continue

            print(
                f"  Received {len(result):,} rows."
            )

            if existing.empty:
                existing = result
            else:
                existing = pd.concat(
                    [
                        existing,
                        result
                    ],
                    ignore_index=True
                )

            output, existing = save_year(
                year,
                existing
            )

            locations_now = len(
                set(
                    zip(
                        existing["weather_latitude"],
                        existing["weather_longitude"]
                    )
                )
            )

            print(
                f"  Saved: {output}"
            )

            print(
                f"  Total rows: {len(existing):,}"
            )

            print(
                f"  Locations completed: "
                f"{locations_now}/{len(required)}"
            )

            time.sleep(
                DELAY_SECONDS
            )

        print(
            f"\nYEAR {year} FINISHED FOR THIS RUN."
        )

    print("\n" + "=" * 70)
    print("WEATHER REPAIR RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
