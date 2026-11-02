import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


DB_NAME = "traffic_accidents"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

INPUT_FILE = "data/weather/staging/weather_daily_staging.csv"


def get_connection():
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        password = input("Enter PostgreSQL password: ")

    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT
    )


def main():

    print("=" * 70)
    print("LOAD OPEN-METEO WEATHER INTO POSTGRESQL")
    print("=" * 70)

    print("\nReading weather staging file...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"Weather staging rows: {len(df):,}")

    # Keep only columns required by PostgreSQL dimension
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
        "wind_speed_10m_max"
    ]

    df = df[columns].copy()

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).dt.date

    df = df.rename(
        columns={"date": "weather_date"}
    )
    # Convert numeric fields
    numeric_columns = [
        "weather_latitude",
        "weather_longitude",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "weather_code",
        "wind_speed_10m_max"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid records
    before = len(df)

    df = df.dropna(
        subset=[
            "weather_date",
            "weather_latitude",
            "weather_longitude"
        ]
    )

    print(
        f"Invalid weather records removed: "
        f"{before - len(df):,}"
    )

    # Remove duplicates
    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "weather_date",
            "weather_latitude",
            "weather_longitude"
        ]
    )

    print(
        f"Duplicate weather records removed: "
        f"{before - len(df):,}"
    )

    print(f"Final records to load: {len(df):,}")

    print("\nConnecting to PostgreSQL...")

    conn = get_connection()

    try:
        cursor = conn.cursor()

        # Clear only the Open-Meteo dimension.
        # Existing accident tables are NOT touched.
        cursor.execute(
            "TRUNCATE TABLE dimensions.dim_open_meteo_weather "
            "RESTART IDENTITY;"
        )

        insert_sql = """
            INSERT INTO dimensions.dim_open_meteo_weather (
                weather_date,
                weather_latitude,
                weather_longitude,
                temperature_2m_mean,
                temperature_2m_max,
                temperature_2m_min,
                precipitation_sum,
                rain_sum,
                snowfall_sum,
                weather_code,
                weather_label,
                wind_speed_10m_max
            )
            VALUES %s
        """

        records = [
            tuple(row)
            for row in df.itertuples(index=False, name=None)
        ]

        print("\nLoading records...")

        execute_values(
            cursor,
            insert_sql,
            records,
            page_size=5000
        )

        conn.commit()

        print("\nOpen-Meteo weather loaded successfully.")

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM dimensions.dim_open_meteo_weather;
            """
        )

        count = cursor.fetchone()[0]

        print(f"PostgreSQL weather rows: {count:,}")

        cursor.close()

    except Exception as e:

        conn.rollback()

        print("\nERROR:")
        print(e)

        raise

    finally:
        conn.close()

    print("\n" + "=" * 70)
    print("OPEN-METEO POSTGRESQL LOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()