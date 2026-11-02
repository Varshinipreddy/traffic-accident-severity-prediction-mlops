CREATE DATABASE traffic_accident;

-- Connect to traffic_accident, then the Python loader will create:
-- accidents_cleaned
-- accident_mart

-- Optional indexes after loading:
-- CREATE INDEX idx_accident_date ON accidents_cleaned(date);
-- CREATE INDEX idx_accident_severity ON accidents_cleaned(severity);
