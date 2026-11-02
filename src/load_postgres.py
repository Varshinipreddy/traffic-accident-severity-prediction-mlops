import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


CSV_FILE = "data/processed/accident_mart.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "traffic_accidents",
    "user": "postgres",
    "password": "mlops123"
}


def clean_value(value):
    """Convert pandas missing values to PostgreSQL NULL."""

    if pd.isna(value):
        return None

    return value


def main():

    print("=" * 60)
    print("LOADING ACCIDENT MART INTO POSTGRESQL")
    print("=" * 60)

    print("\nReading accident mart...")

    df = pd.read_csv(
        CSV_FILE,
        low_memory=False
    )

    print(f"Rows loaded from CSV: {len(df):,}")

    # --------------------------------------------------------
    # Rename CSV columns to PostgreSQL column names
    # --------------------------------------------------------

    df = df.rename(columns={
        "collision_index": "collision_id",
        "date": "accident_date",
        "time": "accident_time",
        "datetime": "accident_datetime"
    })

    # --------------------------------------------------------
    # Convert date/time fields
    # --------------------------------------------------------

    df["accident_date"] = pd.to_datetime(
        df["accident_date"],
        errors="coerce"
    ).dt.date

    df["accident_time"] = pd.to_datetime(
        df["accident_time"],
        format="%H:%M",
        errors="coerce"
    ).dt.time

    df["accident_datetime"] = pd.to_datetime(
        df["accident_datetime"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert boolean columns
    # --------------------------------------------------------

    df["has_fatal_casualty"] = (
        df["has_fatal_casualty"]
        .fillna(0)
        .astype(bool)
    )

    df["has_serious_casualty"] = (
        df["has_serious_casualty"]
        .fillna(0)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Create PostGIS geometry
    # --------------------------------------------------------

    print("\nPreparing spatial coordinates...")

    # Geometry will be created inside PostgreSQL
    # using longitude and latitude.

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    columns = [
        "collision_id",
        "collision_year",
        "collision_ref_no",
        "location_easting_osgr",
        "location_northing_osgr",
        "longitude",
        "latitude",
        "police_force",
        "collision_severity",
        "severity_label",
        "number_of_vehicles",
        "number_of_casualties",
        "accident_date",
        "day_of_week",
        "accident_time",
        "accident_datetime",
        "local_authority_district",
        "local_authority_ons_district",
        "local_authority_highway",
        "local_authority_highway_current",
        "first_road_class",
        "first_road_number",
        "road_type",
        "road_type_label",
        "speed_limit",
        "junction_detail_historic",
        "junction_detail",
        "junction_control",
        "second_road_class",
        "second_road_number",
        "pedestrian_crossing_human_control_historic",
        "pedestrian_crossing_physical_facilities_historic",
        "pedestrian_crossing",
        "light_conditions",
        "light_conditions_label",
        "weather_conditions",
        "weather_label",
        "road_surface_conditions",
        "road_surface_label",
        "special_conditions_at_site",
        "carriageway_hazards_historic",
        "carriageway_hazards",
        "urban_or_rural_area",
        "urban_rural_label",
        "did_police_officer_attend_scene_of_accident",
        "trunk_road_flag",
        "lsoa_of_accident_location",
        "enhanced_severity_collision",
        "collision_injury_based",
        "collision_adjusted_severity_serious",
        "collision_adjusted_severity_slight",
        "year",
        "month",
        "month_name",
        "day",
        "weekday",
        "hour",
        "vehicle_count",
        "average_driver_age",
        "female_driver_count",
        "male_driver_count",
        "escooter_vehicle_count",
        "casualty_count",
        "average_casualty_age",
        "fatal_casualties",
        "serious_casualties",
        "slight_casualties",
        "has_fatal_casualty",
        "has_serious_casualty"
    ]

    # --------------------------------------------------------
    # Connect to PostgreSQL
    # --------------------------------------------------------

    print("\nConnecting to PostgreSQL...")

    conn = psycopg2.connect(**DB_CONFIG)

    cursor = conn.cursor()

    print("✓ PostgreSQL connection successful.")

    # --------------------------------------------------------
    # Prepare INSERT
    # --------------------------------------------------------

    insert_columns = ", ".join(columns)

    placeholders = ", ".join(
        ["%s"] * len(columns)
    )

    insert_sql = f"""
        INSERT INTO fact.fact_collision
        ({insert_columns})
        VALUES ({placeholders})
    """

    # --------------------------------------------------------
    # Insert in batches
    # --------------------------------------------------------

    batch_size = 5000

    cursor.execute("TRUNCATE TABLE fact.fact_collision CASCADE;")
    conn.commit()

    print("\nInserting records...")

    records = []

    for index, row in df.iterrows():

        record = tuple(
            clean_value(row[column])
            for column in columns
        )

        records.append(record)

        if len(records) >= batch_size:

            execute_values(
                cursor,
                insert_sql.replace(
                    "VALUES (" + placeholders + ")",
                    "VALUES %s"
                ),
                records
            )

            conn.commit()

            print(
                f"Inserted: {index + 1:,} / {len(df):,}"
            )

            records = []

    # Insert remaining records

    if records:

        execute_values(
            cursor,
            insert_sql.replace(
                "VALUES (" + placeholders + ")",
                "VALUES %s"
            ),
            records
        )

        conn.commit()

    print("\nCreating PostGIS geometry...")

    cursor.execute("""
        UPDATE fact.fact_collision
        SET geom = ST_SetSRID(
            ST_MakePoint(longitude, latitude),
            4326
        )
        WHERE longitude IS NOT NULL
          AND latitude IS NOT NULL;
    """)

    conn.commit()

    # --------------------------------------------------------
    # Restore star-schema dimension keys
    # --------------------------------------------------------

    print("\nPopulating dimension keys...")

    # Date dimension
    cursor.execute("""
        UPDATE fact.fact_collision f
        SET date_key = d.date_key
        FROM dimensions.dim_date d
        WHERE f.accident_date = d.full_date;
    """)

    # Location dimension
    cursor.execute("""
        UPDATE fact.fact_collision f
        SET location_key = d.location_key
        FROM dimensions.dim_location d
        WHERE f.longitude = d.longitude
          AND f.latitude = d.latitude
          AND COALESCE(f.police_force, -1)
              = COALESCE(d.police_force, -1)
          AND COALESCE(f.local_authority_district, -1)
              = COALESCE(d.local_authority_district, -1)
          AND COALESCE(f.local_authority_ons_district, '')
              = COALESCE(d.local_authority_ons_district, '')
          AND COALESCE(f.lsoa_of_accident_location, '')
              = COALESCE(d.lsoa_of_accident_location, '');
    """)

    # Road dimension
    cursor.execute("""
        UPDATE fact.fact_collision f
        SET road_key = d.road_key
        FROM dimensions.dim_road d
        WHERE COALESCE(f.first_road_class, -1)
              = COALESCE(d.first_road_class, -1)
          AND COALESCE(f.first_road_number, -1)
              = COALESCE(d.first_road_number, -1)
          AND COALESCE(f.road_type, -1)
              = COALESCE(d.road_type, -1)
          AND COALESCE(f.speed_limit, -1)
              = COALESCE(d.speed_limit, -1)
          AND COALESCE(f.junction_detail, -1)
              = COALESCE(d.junction_detail, -1)
          AND COALESCE(f.junction_control, -1)
              = COALESCE(d.junction_control, -1)
          AND COALESCE(f.second_road_class, -1)
              = COALESCE(d.second_road_class, -1)
          AND COALESCE(f.second_road_number, -1)
              = COALESCE(d.second_road_number, -1)
          AND COALESCE(f.road_surface_conditions, -1)
              = COALESCE(d.road_surface_conditions, -1);
    """)

    # DfT weather dimension
    cursor.execute("""
        UPDATE fact.fact_collision f
        SET weather_key = d.weather_key
        FROM dimensions.dim_weather d
        WHERE f.weather_conditions = d.weather_conditions
          AND f.light_conditions = d.light_conditions
          AND COALESCE(f.special_conditions_at_site, -1)
              = COALESCE(d.special_conditions_at_site, -1)
          AND COALESCE(f.carriageway_hazards, -1)
              = COALESCE(d.carriageway_hazards, -1);
    """)

    # Open-Meteo weather dimension
    cursor.execute("""
        UPDATE fact.fact_collision f
        SET open_meteo_weather_key = w.open_meteo_weather_key
        FROM dimensions.dim_open_meteo_weather w
        WHERE f.accident_date = w.weather_date
          AND ROUND((f.latitude * 2)::numeric) / 2
              = w.weather_latitude
          AND ROUND((f.longitude * 2)::numeric) / 2
              = w.weather_longitude;
    """)

    conn.commit()

    print("✓ Date dimension keys populated.")
    print("✓ Location dimension keys populated.")
    print("✓ Road dimension keys populated.")
    print("✓ DfT weather dimension keys populated.")
    print("✓ Open-Meteo weather keys populated.")

    cursor.close()
    conn.close()

    print("\n" + "=" * 60)
    print("POSTGRESQL LOAD COMPLETED")
    print("=" * 60)

    print(f"Records inserted: {len(df):,}")


if __name__ == "__main__":
    main()