
import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
import psycopg2

# =========================================================
# CONFIGURATION
# =========================================================
PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / ".env")

ARTIFACTS = PROJECT / "ml" / "artifacts"
MONITORING = PROJECT / "monitoring"
API_URL = os.getenv("PREDICTION_API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Traffic Accident Analytics & MLOps",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Traffic Accident Analytics & MLOps")
st.caption(
    "UK Road Safety + Open-Meteo | Data Engineering | Machine Learning | Monitoring"
)

# =========================================================
# HELPERS
# =========================================================
def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def metric_value(value, digits=3):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def read_prediction_logs():
    path = MONITORING / "prediction_logs.jsonl"
    if not path.exists():
        return pd.DataFrame()

    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(records)


# =========================================================
# POSTGRESQL
# =========================================================
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "traffic_accidents"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5,
    )


@st.cache_data(ttl=300)
def load_data():
    conn = get_connection()

    query = """
        SELECT
            f.collision_id,
            f.accident_date,
            f.accident_time,
            f.collision_year,
            f.month,
            f.month_name,
            f.weekday,
            f.hour,
            f.latitude,
            f.longitude,
            f.severity_label,
            f.road_type_label,
            f.weather_label AS dft_weather_label,
            COALESCE(w.weather_label, 'Weather unavailable')
                AS open_meteo_weather_label,
            CASE
                WHEN f.open_meteo_weather_key IS NOT NULL
                THEN 'Weather matched'
                ELSE 'Weather unavailable'
            END AS weather_match_status,
            f.light_conditions_label,
            f.road_surface_label,
            f.urban_rural_label,
            f.vehicle_count,
            f.casualty_count,
            f.average_driver_age,
            f.average_casualty_age,
            f.fatal_casualties,
            f.serious_casualties,
            f.slight_casualties,
            w.temperature_2m_mean,
            w.temperature_2m_max,
            w.temperature_2m_min,
            w.precipitation_sum,
            w.rain_sum,
            w.snowfall_sum,
            w.wind_speed_10m_max
        FROM fact.fact_collision f
        LEFT JOIN dimensions.dim_open_meteo_weather w
            ON f.open_meteo_weather_key = w.open_meteo_weather_key
        WHERE f.accident_date IS NOT NULL
    """

    return pd.read_sql_query(query, conn)


# =========================================================
# NAVIGATION
# =========================================================
overview_tab, ml_tab, prediction_tab, monitoring_tab = st.tabs(
    [
        "📊 Accident Analytics",
        "🤖 Model Performance",
        "🔮 Prediction Demo",
        "📈 API Monitoring",
    ]
)

# =========================================================
# TAB 1 — PART 1 ACCIDENT ANALYTICS
# =========================================================
with overview_tab:
    try:
        df = load_data()
    except Exception as exc:
        st.error("Could not connect to PostgreSQL.")
        st.code(str(exc))
        st.info(
            "Check PostgreSQL is running and the DB_HOST, DB_PORT, "
            "DB_NAME, DB_USER and DB_PASSWORD values in .env are correct."
        )
        df = pd.DataFrame()

    if df.empty:
        st.warning("No accident records are available to display.")
    else:
        df["accident_date"] = pd.to_datetime(
            df["accident_date"], errors="coerce"
        )

        st.sidebar.header("🔎 Accident Filters")

        years = sorted(df["collision_year"].dropna().unique().tolist())
        severities = sorted(df["severity_label"].dropna().unique().tolist())
        roads = sorted(df["road_type_label"].dropna().unique().tolist())

        selected_years = st.sidebar.multiselect(
            "Year", years, default=years
        )
        selected_severity = st.sidebar.multiselect(
            "Severity", severities, default=severities
        )
        selected_roads = st.sidebar.multiselect(
            "Road Type", roads, default=roads
        )

        filtered = df[
            df["collision_year"].isin(selected_years)
            & df["severity_label"].isin(selected_severity)
            & df["road_type_label"].isin(selected_roads)
        ].copy()

        st.subheader("Key Performance Indicators")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accidents", f"{len(filtered):,}")
        c2.metric(
            "Casualties",
            f"{int(filtered['casualty_count'].fillna(0).sum()):,}",
        )
        c3.metric(
            "Vehicles Involved",
            f"{int(filtered['vehicle_count'].fillna(0).sum()):,}",
        )
        c4.metric(
            "Fatal Accidents",
            f"{int((filtered['severity_label'] == 'Fatal').sum()):,}",
        )

        left, right = st.columns(2)

        with left:
            st.subheader("1. Accident Severity")
            severity = (
                filtered["severity_label"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("severity")
                .reset_index(name="accidents")
            )
            fig = px.bar(
                severity, x="severity", y="accidents",
                title="Accidents by Severity",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("2. Accidents by Hour")
            hourly = (
                filtered.groupby("hour", dropna=False)
                .size().reset_index(name="accidents")
            )
            fig = px.line(
                hourly, x="hour", y="accidents",
                markers=True, title="Accidents by Hour",
            )
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("3. DfT Weather Conditions")
            weather = (
                filtered["dft_weather_label"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("weather")
                .reset_index(name="accidents")
            )
            fig = px.bar(
                weather, x="weather", y="accidents",
                title="Accidents by Reported Weather",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("4. Road Type")
            road = (
                filtered["road_type_label"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("road_type")
                .reset_index(name="accidents")
            )
            fig = px.bar(
                road, x="road_type", y="accidents",
                title="Accidents by Road Type",
            )
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("5. Weekday Analysis")
            weekday_order = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            weekday = (
                filtered.groupby("weekday")
                .size()
                .reindex(weekday_order)
                .fillna(0)
                .rename("accidents")
                .rename_axis("weekday")
                .reset_index()
            )
            fig = px.bar(
                weekday, x="weekday", y="accidents",
                title="Accidents by Day of Week",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("6. Urban vs Rural")
            urban = (
                filtered["urban_rural_label"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("area")
                .reset_index(name="accidents")
            )
            fig = px.pie(
                urban, names="area", values="accidents",
                title="Urban vs Rural Accidents",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("7. Open-Meteo Weather Enrichment")
        weather_available = filtered[
            filtered["weather_match_status"] == "Weather matched"
        ].copy()

        w1, w2, w3 = st.columns(3)
        w1.metric("Weather-Matched Accidents", f"{len(weather_available):,}")
        match_rate = (
            len(weather_available) / len(filtered) * 100
            if len(filtered) else 0
        )
        w2.metric("Weather Match Rate", f"{match_rate:.2f}%")
        w3.metric(
            "Rows with Temperature",
            f"{weather_available['temperature_2m_mean'].notna().sum():,}",
        )

        if not weather_available.empty:
            open_weather = (
                weather_available["open_meteo_weather_label"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("weather")
                .reset_index(name="accidents")
            )
            fig = px.bar(
                open_weather, x="weather", y="accidents",
                title="Accidents by Open-Meteo Weather Category",
            )
            st.plotly_chart(fig, use_container_width=True)

            temperature = weather_available.dropna(
                subset=["temperature_2m_mean", "precipitation_sum"]
            )
            if not temperature.empty:
                sample = temperature.sample(
                    min(10000, len(temperature)), random_state=42
                )
                fig = px.scatter(
                    sample,
                    x="temperature_2m_mean",
                    y="precipitation_sum",
                    color="severity_label",
                    title="Temperature vs Precipitation",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("8. Accident Location Hotspot Map")
        geo = filtered.dropna(subset=["latitude", "longitude"])
        if not geo.empty:
            map_data = geo[["latitude", "longitude"]].sample(
                min(5000, len(geo)), random_state=42
            )
            st.map(map_data, latitude="latitude", longitude="longitude")
        else:
            st.info("No valid coordinates available.")

        st.subheader("9. Data Quality & Weather Coverage")
        q1, q2, q3 = st.columns(3)
        q1.metric("Total Database Records", f"{len(df):,}")
        q2.metric("Filtered Records", f"{len(filtered):,}")
        unavailable = (
            filtered["weather_match_status"] == "Weather unavailable"
        ).sum()
        q3.metric("Weather Unavailable", f"{unavailable:,}")

        coverage = (
            filtered["weather_match_status"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("weather_status")
            .reset_index(name="records")
        )
        fig = px.pie(
            coverage, names="weather_status", values="records",
            title="Open-Meteo Weather Coverage",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download Filtered Accident Data",
            data=filtered.to_csv(index=False),
            file_name="filtered_accident_weather_data.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with ml_tab:
    st.header("🤖 Machine Learning Model Performance")
    st.caption("Metrics loaded from your saved model evaluation artifacts.")

    comparison = read_json(ARTIFACTS / "model_comparison.json")
    report = read_json(ARTIFACTS / "best_model_report.json")

    if not comparison:
        st.warning(
            f"Model comparison file not found: "
            f"{ARTIFACTS / 'model_comparison.json'}"
        )
    else:
        comparison_df = pd.DataFrame(comparison).T
        comparison_df.index.name = "Model"
        comparison_df = comparison_df.reset_index()

        if "macro_f1" in comparison_df.columns:
            comparison_df = comparison_df.sort_values(
                "macro_f1", ascending=False
            )

        st.subheader("Model Comparison")
        display_df = comparison_df.copy()
        for col in [
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
        ]:
            if col in display_df:
                display_df[col] = display_df[col].map(
                    lambda x: round(float(x), 4)
                )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        metric_options = [
            c for c in
            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
            if c in comparison_df.columns
        ]
        if metric_options:
            chosen_metric = st.selectbox(
                "Compare models by metric",
                metric_options,
                index=metric_options.index("macro_f1")
                if "macro_f1" in metric_options else 0,
            )
            fig = px.bar(
                comparison_df,
                x="Model",
                y=chosen_metric,
                title=f"Model Comparison — {chosen_metric}",
                text_auto=".3f",
            )
            st.plotly_chart(fig, use_container_width=True)

        if "training_seconds" in comparison_df.columns:
            fig = px.bar(
                comparison_df,
                x="Model",
                y="training_seconds",
                title="Training Time (seconds)",
                text_auto=".1f",
            )
            st.plotly_chart(fig, use_container_width=True)

        best_model = (
            comparison_df.sort_values("macro_f1", ascending=False)
            .iloc[0]["Model"]
            if "macro_f1" in comparison_df.columns else "Not available"
        )

        st.info(
            f"Selected by macro F1: **{best_model}**. "
            "Macro F1 gives equal weight to each severity class."
        )

    st.subheader("Best Model: Class-wise Evaluation")
    if report:
        class_rows = []
        for label in ["Fatal", "Serious", "Slight"]:
            if label in report:
                class_rows.append({
                    "Severity": label,
                    "Precision": report[label].get("precision"),
                    "Recall": report[label].get("recall"),
                    "F1 score": report[label].get("f1-score"),
                    "Test support": report[label].get("support"),
                })

        if class_rows:
            class_df = pd.DataFrame(class_rows)
            st.dataframe(
                class_df.style.format({
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1 score": "{:.3f}",
                    "Test support": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            fig = px.bar(
                class_df.melt(
                    id_vars="Severity",
                    value_vars=["Precision", "Recall", "F1 score"],
                    var_name="Metric",
                    value_name="Score",
                ),
                x="Severity", y="Score", color="Metric",
                barmode="group",
                title="Precision, Recall and F1 by Severity",
                range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

            fatal = next(
                (r for r in class_rows if r["Severity"] == "Fatal"), None
            )
            if fatal:
                st.warning(
                    f"Safety limitation: Fatal precision is "
                    f"{float(fatal['Precision']):.1%} and fatal recall is "
                    f"{float(fatal['Recall']):.1%}. The model produces "
                    "many false fatal alerts and misses some fatal cases. "
                    "Do not use this model for real-world safety decisions."
                )
    else:
        st.warning("Class-wise report JSON is not available.")

    st.markdown("### Interpretation")
    st.write(
        "- Accuracy alone can be misleading when severity classes are imbalanced.\n"
        "- Macro F1 and balanced accuracy help assess performance across classes.\n"
        "- Fatal cases are rare, so fatal precision and recall need particular attention.\n"
        "- These metrics describe a held-out evaluation set; they do not guarantee "
        "performance on future accidents."
    )

# =========================================================
# TAB 3 — PREDICTION API
# =========================================================
with prediction_tab:
    st.subheader("Predict Accident Severity")
    st.write(
        "Enter the accident conditions below. The API will estimate the "
        "predicted severity."
    )

    sample_path = PROJECT / "ml" / "artifacts" / "train.csv"

    if not sample_path.exists():
        st.error(f"Training sample not found: {sample_path}")
    else:
        sample_df = pd.read_csv(sample_path, nrows=5)

        # Use a complete feature row as the base payload.
        excluded_cols = {
            "collision_severity",
            "severity_label",
            "predicted_severity",
        }
        feature_cols = [
            c for c in sample_df.columns
            if c not in excluded_cols
        ]
        base_payload = sample_df.iloc[0][feature_cols].to_dict()

        def clean_value(value):
            if pd.isna(value):
                return None
            if hasattr(value, "item"):
                return value.item()
            return value

        base_payload = {
            key: clean_value(value)
            for key, value in base_payload.items()
        }

        # Match fields that actually exist in the training data.
        def find_column(possible_names):
            lookup = {
                str(col).strip().lower().replace(" ", "_"): col
                for col in feature_cols
            }
            for name in possible_names:
                normalized = name.strip().lower().replace(" ", "_")
                if normalized in lookup:
                    return lookup[normalized]
            return None

        field_specs = [
            (
                "Speed limit",
                ["speed_limit"],
                "number",
                "Enter the speed limit."
            ),
            (
                "Number of vehicles",
                ["number_of_vehicles", "num_vehicles"],
                "number",
                "Vehicles involved in the collision."
            ),
            (
                "Number of casualties",
                ["number_of_casualties", "num_casualties"],
                "number",
                "Casualties recorded for the collision."
            ),
            (
                "Road type",
                ["road_type"],
                "category",
                "Select the road type."
            ),
            (
                "Weather conditions",
                ["weather_conditions", "weather_condition"],
                "category",
                "Select the recorded weather condition."
            ),
            (
                "Road surface",
                ["road_surface_conditions", "road_surface_condition"],
                "category",
                "Select the recorded road surface."
            ),
            (
                "Light conditions",
                ["light_conditions", "light_condition"],
                "category",
                "Select the recorded lighting."
            ),
            (
                "Urban or rural area",
                ["urban_or_rural_area", "urban_rural"],
                "category",
                "Select the area type."
            ),
            (
                "Day of week",
                ["day_of_week"],
                "category",
                "Select the day of the collision."
            ),
            (
                "Latitude",
                ["latitude"],
                "number",
                "Approximate latitude."
            ),
            (
                "Longitude",
                ["longitude"],
                "number",
                "Approximate longitude."
            ),
        ]

        matched_fields = []
        for label, aliases, field_type, help_text in field_specs:
            column = find_column(aliases)
            if column is not None:
                matched_fields.append(
                    (label, column, field_type, help_text)
                )

        with st.form("accident_prediction_form"):
            st.caption(
                "Fields are shown when their matching columns exist "
                "in your training dataset. Other model inputs use a "
                "sample training row."
            )

            edited_values = {}

            col_left, col_right = st.columns(2)

            for index, (label, column, field_type, help_text) in enumerate(
                matched_fields
            ):
                container = col_left if index % 2 == 0 else col_right
                original_value = base_payload.get(column)

                with container:
                    if field_type == "category":
                        options = (
                            sample_df[column]
                            .dropna()
                            .astype(str)
                            .value_counts()
                            .index
                            .tolist()
                        )

                        if original_value is not None:
                            original_text = str(original_value)
                            if original_text not in options:
                                options.insert(0, original_text)

                        if options:
                            default_index = (
                                options.index(str(original_value))
                                if original_value is not None
                                and str(original_value) in options
                                else 0
                            )
                            selected = st.selectbox(
                                label,
                                options,
                                index=default_index,
                                help=help_text,
                                key=f"prediction_{column}",
                            )
                            edited_values[column] = selected

                    else:
                        try:
                            default_number = float(original_value)
                        except (TypeError, ValueError):
                            default_number = 0.0

                        entered_number = st.number_input(
                            label,
                            value=default_number,
                            help=help_text,
                            key=f"prediction_{column}",
                        )
                        edited_values[column] = entered_number

            submitted = st.form_submit_button(
                "Predict severity",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            payload = base_payload.copy()
            payload.update(edited_values)

            # Send JSON-compatible values to the existing API.
            payload = {
                key: clean_value(value)
                for key, value in payload.items()
            }

            try:
                health_response = requests.get(
                    "http://localhost:8001/health",
                    timeout=5,
                )
                health_response.raise_for_status()

                prediction_response = requests.post(
                    "http://localhost:8001/predict",
                    json=payload,
                    timeout=30,
                )
                prediction_response.raise_for_status()
                result = prediction_response.json()

                st.success("Prediction received from the API.")

                severity = result.get("severity_label", "Unknown")
                st.metric("Predicted severity", severity)

                st.json(result)

            except requests.exceptions.RequestException as exc:
                st.error(
                    "Could not get a prediction. Check that the API "
                    "is running on http://localhost:8001 and try again."
                )
                st.caption(f"Technical details: {exc}")

# =========================================================
# TAB 4 — MONITORING
# =========================================================
with monitoring_tab:
    st.header("📈 Prediction API Monitoring")
    st.caption(
        "Metrics are calculated from monitoring/prediction_logs.jsonl. "
        "They reflect only requests recorded by the logging code."
    )

    logs = read_prediction_logs()

    if logs.empty:
        st.info(
            "No prediction logs found yet. Make a prediction using the API "
            "to create monitoring data."
        )
    else:
        if "success" in logs:
            success_series = logs["success"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            )
            failures = int((~success_series).sum())
        else:
            success_series = pd.Series(
                [True] * len(logs), index=logs.index
            )
            failures = 0

        total = len(logs)
        failure_rate = failures / total * 100 if total else 0

        latency = (
            pd.to_numeric(logs["latency_ms"], errors="coerce").dropna()
            if "latency_ms" in logs else pd.Series(dtype=float)
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Logged Requests", f"{total:,}")
        m2.metric("Failed Requests", f"{failures:,}")
        m3.metric("Failure Rate", f"{failure_rate:.2f}%")
        m4.metric(
            "Average Latency",
            f"{latency.mean():.1f} ms" if not latency.empty else "N/A",
        )

        if not latency.empty:
            p95 = latency.quantile(0.95)
            st.metric("P95 Latency", f"{p95:.1f} ms")
            fig = px.histogram(
                latency.to_frame(name="latency_ms"),
                x="latency_ms",
                title="Distribution of API Latency",
                nbins=20,
            )
            st.plotly_chart(fig, use_container_width=True)

        if "timestamp" in logs:
            logs["timestamp"] = pd.to_datetime(
                logs["timestamp"], errors="coerce", utc=True
            )
            valid_times = logs.dropna(subset=["timestamp"])
            if not valid_times.empty:
                requests_over_time = (
                    valid_times.set_index("timestamp")
                    .resample("h").size()
                    .reset_index(name="requests")
                )
                fig = px.line(
                    requests_over_time,
                    x="timestamp", y="requests",
                    markers=True,
                    title="Logged Requests Over Time",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Prediction Logs")
        st.dataframe(logs.tail(100), use_container_width=True)

        if "actual_severity" not in logs.columns:
            st.info(
                "False-negative rate is not available because actual outcome "
                "labels are not currently recorded alongside predictions. "
                "This is expected for unlabeled live requests."
            )

        st.download_button(
            "Download Monitoring Logs",
            data=logs.to_csv(index=False),
            file_name="prediction_monitoring_logs.csv",
            mime="text/csv",
        )

    st.subheader("Suggested Monitoring and Retraining Policy")
    st.markdown(
        """
        These are **proposed thresholds**, not measured alerts:

        - Investigate if API failure rate exceeds 5% over a meaningful window.
        - Investigate if P95 latency exceeds the service-level target.
        - Re-evaluate if macro F1 falls more than 10% relative to the approved baseline.
        - Review fatal-class recall and false negatives when ground-truth outcomes arrive.
        - Evaluate feature drift using new input data against the training distribution.
        - Retrain only after reviewing data quality, drift, and performance on a fresh
          time-based validation set.
        """
    )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Traffic Accident Severity Analytics & MLOps | Academic prototype | "
    "Predictions are experimental and must not be used as real-world safety advice."
)
