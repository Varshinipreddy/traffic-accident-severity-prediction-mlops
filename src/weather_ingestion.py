import os
import time
import logging
from datetime import datetime

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "accident_mart.csv",
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "weather",
    "raw",
)

STAGING_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "weather",
    "staging",
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs",
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# WEATHER GRID
# ============================================================

# 0.5 degree grid gives us 196 cells for this project.
GRID_SIZE = 0.5

# Keep API responses reasonably sized.
LOCATION_BATCH_SIZE = 10

# Delay between successful API requests.
REQUEST_DELAY_SECONDS = 3

# Retry delays for temporary API throttling.
RETRY_DELAYS = [60, 120, 300]


# ============================================================
# OPEN-METEO VARIABLES
# ============================================================

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


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = os.path.join(
    LOG_DIR,
    "weather_ingestion.log",
)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD ACCIDENT DATA
# ============================================================

def load_accident_locations():
    logger.info("Loading accident mart")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "collision_index",
            "datetime",
            "latitude",
            "longitude",
        ],
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "latitude",
            "longitude",
        ]
    ).copy()

    # Create the weather grid.
    df["weather_latitude"] = (
        (df["latitude"] / GRID_SIZE).round() * GRID_SIZE
    )

    df["weather_longitude"] = (
        (df["longitude"] / GRID_SIZE).round() * GRID_SIZE
    )

    logger.info(
        "Valid accident records: %s",
        len(df),
    )

    logger.info(
        "Weather grid locations: %s",
        df[
            [
                "weather_latitude",
                "weather_longitude",
            ]
        ].drop_duplicates().shape[0],
    )

    return df


# ============================================================
# OPEN-METEO REQUEST
# ============================================================

def request_weather(
    locations,
    start_date,
    end_date,
):
    latitudes = ",".join(
        str(round(lat, 2))
        for lat, lon in locations
    )

    longitudes = ",".join(
        str(round(lon, 2))
        for lat, lon in locations
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "GMT",
        "wind_speed_unit": "kmh",
    }

    for attempt in range(len(RETRY_DELAYS) + 1):

        try:

            logger.info(
                "Requesting weather: %s to %s | locations=%s",
                start_date,
                end_date,
                len(locations),
            )

            response = requests.get(
                API_URL,
                params=params,
                timeout=120,
            )

            # Explicit handling of API throttling.
            if response.status_code == 429:

                if attempt < len(RETRY_DELAYS):

                    wait_time = RETRY_DELAYS[attempt]

                    logger.warning(
                        "Open-Meteo returned 429. "
                        "Waiting %s seconds before retry.",
                        wait_time,
                    )

                    time.sleep(wait_time)

                    continue

                logger.error(
                    "Open-Meteo still returning 429 "
                    "after retries."
                )

                return []

            response.raise_for_status()

            data = response.json()

            # Open-Meteo returns a dictionary for one location
            # and a list for multiple locations.
            if isinstance(data, dict):
                data = [data]

            results = []

            for location, weather_data in zip(
                locations,
                data,
            ):

                if "daily" not in weather_data:
                    logger.warning(
                        "No daily data returned for location %s",
                        location,
                    )
                    continue

                daily = weather_data["daily"]

                dates = daily.get("time", [])

                for i, date_value in enumerate(dates):

                    row = {
                        "weather_latitude": location[0],
                        "weather_longitude": location[1],
                        "weather_date": date_value,
                    }

                    for variable in DAILY_VARIABLES:

                        values = daily.get(
                            variable,
                            [],
                        )

                        if i < len(values):
                            row[variable] = values[i]
                        else:
                            row[variable] = None

                    results.append(row)

            return results

        except requests.exceptions.RequestException as exc:

            logger.error(
                "Weather API request failed: %s",
                exc,
            )

            if attempt < len(RETRY_DELAYS):

                wait_time = RETRY_DELAYS[attempt]

                logger.warning(
                    "Retrying after %s seconds",
                    wait_time,
                )

                time.sleep(wait_time)

            else:
                return []

    return []


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OPEN-METEO WEATHER INGESTION")
    print("=" * 70)

    accidents = load_accident_locations()

    locations = (
        accidents[
            [
                "weather_latitude",
                "weather_longitude",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "weather_latitude",
                "weather_longitude",
            ]
        )
    )

    locations = list(
        locations.itertuples(
            index=False,
            name=None,
        )
    )

    start_date = accidents["datetime"].min().date()
    end_date = accidents["datetime"].max().date()

    print()
    print(f"Accident date range: {start_date} to {end_date}")
    print(f"Weather grid locations: {len(locations)}")
    print()

    logger.info(
        "Weather extraction started: %s to %s",
        start_date,
        end_date,
    )

    all_weather = []

    # Process one calendar year at a time.
    for year in range(
        start_date.year,
        end_date.year + 1,
    ):

        year_start = max(
            start_date,
            datetime(year, 1, 1).date(),
        )

        year_end = min(
            end_date,
            datetime(year, 12, 31).date(),
        )

        print(
            f"\nProcessing year {year}: "
            f"{year_start} to {year_end}"
        )

        logger.info(
            "Processing year %s",
            year,
        )

        yearly_weather = []

        # Process only 10 locations per API request.
        for batch_start in range(
            0,
            len(locations),
            LOCATION_BATCH_SIZE,
        ):

            batch = locations[
                batch_start:
                batch_start + LOCATION_BATCH_SIZE
            ]

            batch_number = (
                batch_start // LOCATION_BATCH_SIZE
            ) + 1

            total_batches = (
                (len(locations) + LOCATION_BATCH_SIZE - 1)
                // LOCATION_BATCH_SIZE
            )

            print(
                f"  Batch {batch_number}/{total_batches} "
                f"({len(batch)} locations)"
            )

            data = request_weather(
                batch,
                str(year_start),
                str(year_end),
            )

            if data:

                yearly_weather.extend(data)

                print(
                    f"    Received {len(data):,} rows"
                )

            else:

                print(
                    "    No data returned for this batch"
                )

            # Give the API breathing room.
            time.sleep(
                REQUEST_DELAY_SECONDS
            )

        if yearly_weather:

            year_df = pd.DataFrame(
                yearly_weather
            )

            year_file = os.path.join(
                RAW_DIR,
                f"open_meteo_daily_weather_{year}.csv",
            )

            year_df.to_csv(
                year_file,
                index=False,
            )

            yearly_weather = year_df

            print(
                f"  Saved: {year_file}"
            )

            print(
                f"  Rows: {len(year_df):,}"
            )

            logger.info(
                "Saved %s weather rows for %s",
                len(year_df),
                year,
            )

            all_weather.append(
                year_df
            )

    # ========================================================
    # COMBINE YEARS
    # ========================================================

    if not all_weather:

        raise RuntimeError(
            "No weather data was retrieved."
        )

    weather = pd.concat(
        all_weather,
        ignore_index=True,
    )

    weather["weather_date"] = pd.to_datetime(
        weather["weather_date"],
        errors="coerce",
    ).dt.date

    weather = weather.dropna(
        subset=["weather_date"]
    )

    # Remove duplicate weather observations.
    weather = weather.drop_duplicates(
        subset=[
            "weather_latitude",
            "weather_longitude",
            "weather_date",
        ]
    )

    # ========================================================
    # DATA TYPES
    # ========================================================

    numeric_columns = [
        column
        for column in DAILY_VARIABLES
        if column != "weather_code"
    ]

    for column in numeric_columns:

        weather[column] = pd.to_numeric(
            weather[column],
            errors="coerce",
        )

    weather["weather_code"] = pd.to_numeric(
        weather["weather_code"],
        errors="coerce",
    )

    # ========================================================
    # SAVE STAGING FILE
    # ========================================================

    staging_file = os.path.join(
        STAGING_DIR,
        "weather_daily_staging.csv",
    )

    weather.to_csv(
        staging_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("WEATHER INGESTION COMPLETE")
    print("=" * 70)
    print(f"Total weather rows: {len(weather):,}")
    print(f"Weather locations: {weather[['weather_latitude', 'weather_longitude']].drop_duplicates().shape[0]}")
    print(f"Date range: {weather['weather_date'].min()} to {weather['weather_date'].max()}")
    print(f"Staging file: {staging_file}")
    print("=" * 70)

    logger.info(
        "Weather ingestion completed successfully"
    )


if __name__ == "__main__":
    main()