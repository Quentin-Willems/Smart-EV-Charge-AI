-- postgres_data_wrangler.sql
-- Author: Quentin Willems
-- Project: Smart EV Charge recommendations leveraging AI and Elia open data
-- Purpose: Clean and transform raw datasets into structured tables for analytics
-- Database: Raw data are stored in an external PostgreSQL database
-- Open Data Source: https://www.elia.be/fr/donnees-de-reseau/open-data

-- ------------------------
-- 1. Imbalance price    --
-- ------------------------

-- Drop the table if it already exists to make the script rerunnable.
DROP TABLE IF EXISTS imbalance_price_clean;

-- Create a clean dataset with hourly averages from the raw data.
CREATE TABLE imbalance_price_clean AS
SELECT 
    date_trunc('hour', "DateTime"::timestamp) AS timestamp,
    EXTRACT(YEAR FROM "DateTime"::timestamp) AS year,
    TO_CHAR("DateTime"::timestamp, 'MM') AS month,
-- `DOW` returns a value from 0 (Sunday) to 6 (Saturday), which is a common standard.
    EXTRACT(DOW FROM "DateTime"::timestamp) AS weekday,
    TO_CHAR("DateTime"::timestamp, 'HH24') AS hour,
    TO_CHAR("DateTime"::timestamp, 'WW') AS calendar_week,

    AVG("System imbalance") AS avg_systemimbalance,  -- Imbalance itself as observed
    AVG("Net regulation volume") AS avg_netregulationvolume, -- Transmission System Operator reaction to the observed imbalance
    AVG("Positive imbalance price") AS avg_positive_price,
    AVG("Negative imbalance price") AS avg_negative_price
FROM 
    imbalance_prices_per_quarter_hour
WHERE 
    "Quality status" IN ('Validated', 'Non-validated')
GROUP BY 
    date_trunc('hour', "DateTime"::timestamp),
    EXTRACT(YEAR FROM "DateTime"::timestamp),
    TO_CHAR("DateTime"::timestamp, 'MM'),
    EXTRACT(DOW FROM "DateTime"::timestamp),
    TO_CHAR("DateTime"::timestamp, 'HH24'),
    TO_CHAR("DateTime"::timestamp, 'WW');

-- -------------------------------------
-- 2. Photovoltaic (PV) production    --
-- -------------------------------------
 
----- START: pivot_pv_production() function ------

-- pivot_pv_production() is a stored function designed to automate the data pivoting process for each region.
-- Instead of manually writing a column for each region, the function handles it dynamically by adapting to all regions present in the source table.
CREATE OR REPLACE FUNCTION pivot_pv_production()
-- RETURN void specifies that the function does not return any value. Its purpose is to perform an action (creating a table) rather than calculating and returning a result.
RETURNS void AS $$

-- Variable Declaration
DECLARE
    -- Variable to store the complete dynamic CREATE TABLE SQL query
    sql_query text;
    -- Variable to temporary store region names during the loop iteration
    region_name text;
    -- Variable for the dynamic SELECT clause (A string that will be incrementally built to contain all the conditional aggregation columns (the AVG(CASE WHEN ...) expressions).
    select_clause text := '';
    -- Variable for the static GROUP BY clause
    group_by_clause text;

-- Note: in plpgsql BEGIN and END define a block of executable code. 
BEGIN
    -- Step 1: Looping through Regions
    -- Get unique region names for the pivot and build the SELECT part of the query for each region.
    -- Note: Add a Replace function to replace any ' ' or '-' in the name of the Region by '_' to avoid issue in the query
    FOR region_name IN SELECT DISTINCT "Region" FROM pv_production_belgian_grid ORDER BY "Region" LOOP
        -- Add columns for `avg_measured_upscaled` for each region.
        -- In each iteration, we append a new part to the select_clause variable using the format() function in plpgsql
        select_clause := select_clause || format(
            ', AVG(CASE WHEN "Region" = %L THEN "Measured & Upscaled" ELSE NULL END) AS avg_pv_measured_upscaled_%s', 
            region_name, 
            REPLACE(REPLACE(lower(region_name), ' ', '_'), '-', '_')
        );

        -- Add columns for `avg_monitored_capacity` for each region.
        select_clause := select_clause || format(
            ', AVG(CASE WHEN "Region" = %L THEN "Monitored capacity" ELSE NULL END) AS avg_pv_monitored_capacity_%s', 
            region_name, 
            REPLACE(REPLACE(lower(region_name), ' ', '_'), '-', '_')
        );
    END LOOP;

    -- Step 2: Build the GROUP BY clause.
    -- We group by time-related columns only, as the pivot is done through aggregation.
    group_by_clause := 'GROUP BY 
        date_trunc(''hour'', "Datetime"::timestamp), 
        EXTRACT(DOW FROM "Datetime"::timestamp), 
        EXTRACT(YEAR FROM "Datetime"::timestamp),
        TO_CHAR("Datetime"::timestamp, ''HH24''), 
        TO_CHAR("Datetime"::timestamp, ''MM''), 
        TO_CHAR("Datetime"::timestamp, ''WW'')';

    -- Step 3: Build the complete final query.
    sql_query := format('
        DROP TABLE IF EXISTS pv_production_clean;

        CREATE TABLE pv_production_clean AS
        SELECT 
            date_trunc(''hour'', "Datetime"::timestamp) AS timestamp,
            EXTRACT(DOW FROM "Datetime"::timestamp) AS weekday,
            EXTRACT(YEAR FROM "Datetime"::timestamp) AS year,
            TO_CHAR("Datetime"::timestamp, ''HH24'') AS hour,
            TO_CHAR("Datetime"::timestamp, ''MM'') AS month,
            TO_CHAR("Datetime"::timestamp, ''WW'') AS calendar_week
            %s
        FROM 
            pv_production_belgian_grid
        WHERE 
            "Measured & Upscaled" IS NOT NULL
        %s;', 
        select_clause, 
        group_by_clause
    );

    -- Step 4: Execute the dynamic query.
    EXECUTE sql_query;
    
END;
$$ LANGUAGE plpgsql;
----- END: pivot_pv_production() function ------

-- Call the pivot_pv_production()  function to create the pivoted table pv_production_clean
SELECT pivot_pv_production();




-- ------------------------
-- 3. Wind production    --
-- ------------------------
----- START: pivot_wind_production() function ------

-- Same Logic as pivot_pv_production() function above --
-- Note: data issue in data source for "Measured & Upscaled" which is not numeric in this table
--       So in this "Measured & Upscaled" is cast to numeric to allow dynamic input reloading

CREATE OR REPLACE FUNCTION pivot_wind_production()
RETURNS void AS $$

DECLARE
    sql_query text;
    region_name text;
    clean_region_suffix text;
    select_clause text := '';
    group_by_clause text;

BEGIN
    -- Step 1 : Loop through regions
    
    FOR region_name IN SELECT DISTINCT "Region" FROM wind_power_production_estimation_and_forecast_on_belgian_grid WHERE "Region" IS NOT NULL ORDER BY "Region" LOOP
        clean_region_suffix := REPLACE(REPLACE(lower(region_name), ' ', '_'), '-', '_');
        select_clause := select_clause || format(
            ', AVG(CASE WHEN "Region" = %L THEN "Measured & Upscaled"::numeric ELSE NULL END) AS %I',
            region_name,
            'avg_wind_measured_' || clean_region_suffix
        );


        select_clause := select_clause || format(
            ', AVG(CASE WHEN "Region" = %L THEN "Monitored capacity" ELSE NULL END) AS %I',
            region_name,
            'avg_wind_monitored_capacity_' || clean_region_suffix
        );
    END LOOP;

    -- Step 2 : Built static group_by clause
 
    group_by_clause := 'GROUP BY 
        date_trunc(''hour'', "Datetime"::timestamp), 
        EXTRACT(DOW FROM "Datetime"::timestamp), 
        EXTRACT(YEAR FROM "Datetime"::timestamp),
        TO_CHAR("Datetime"::timestamp, ''HH24''), 
        TO_CHAR("Datetime"::timestamp, ''MM''), 
        TO_CHAR("Datetime"::timestamp, ''WW'')';

    -- Step 3 : Built final dynamic query
    -- Note: WHERE claise filters the data to include only rows where the "Measured & Upscaled" column contains a value that is a valid number (integer or decimal, positive or negative) after removing any surrounding spaces. 
    --       This prevents errors when the next part of the query tries to cast the value to a numeric type.
    sql_query := format('
        DROP TABLE IF EXISTS wind_production_clean;

        CREATE TABLE wind_production_clean AS
        SELECT 
            date_trunc(''hour'', "Datetime"::timestamp) AS timestamp,
            EXTRACT(DOW FROM "Datetime"::timestamp) AS weekday,
            EXTRACT(YEAR FROM "Datetime"::timestamp) AS year,
            TO_CHAR("Datetime"::timestamp, ''HH24'') AS hour,
            TO_CHAR("Datetime"::timestamp, ''MM'') AS month,
            TO_CHAR("Datetime"::timestamp, ''WW'') AS calendar_week
            %s
        FROM 
            wind_power_production_estimation_and_forecast_on_belgian_grid
        WHERE 
            TRIM("Measured & Upscaled") ~ ''^-?\d+(\.\d+)?$''
        %s;', 
        select_clause, 
        group_by_clause
    );

    -- Step 4 : Execute dynamic query
    EXECUTE sql_query;
    
END;
$$ LANGUAGE plpgsql;
----- END: pivot_wind_production() function ------

-- Call the function pivot_wind_production()
SELECT pivot_wind_production();


-- ------------------------
-- 4. Grid Load          --
-- ------------------------

DROP TABLE IF EXISTS total_load_clean;

CREATE TABLE total_load_clean AS
SELECT 
    date_trunc('hour', "Datetime"::timestamp) AS timestamp,
    EXTRACT(DOW FROM "Datetime"::timestamp) AS weekday,
    EXTRACT(YEAR FROM "Datetime"::timestamp) AS year,
    TO_CHAR("Datetime"::timestamp, 'HH24') AS hour,
    TO_CHAR("Datetime"::timestamp, 'MM') AS month,
    TO_CHAR("Datetime"::timestamp, 'WW') AS calendar_week,
    AVG("Total Load") AS avg_load 
FROM 
    measured_and_forecasted_total_load_on_the_belgian_grid
WHERE 
    "Total Load" IS NOT NULL
GROUP BY 
    date_trunc('hour', "Datetime"::timestamp),
    EXTRACT(DOW FROM "Datetime"::timestamp),
    EXTRACT(YEAR FROM "Datetime"::timestamp),
    TO_CHAR("Datetime"::timestamp, 'HH24'),
    TO_CHAR("Datetime"::timestamp, 'MM'),
    TO_CHAR("Datetime"::timestamp, 'WW');




-- ------------------------
-- 5. Master table       --
-- ------------------------



-- Purpose: Creates a master table for AI/ML time series forecasting by joining cleaned tables.

DROP TABLE IF EXISTS master_data;

CREATE TABLE master_data AS
SELECT 
    p.timestamp,
    p.year,
    p.month,
    p.weekday,
    p.hour,
    p.calendar_week,
    p.avg_pv_measured_upscaled_antwerp AS Area_Antwerp_pv_production,
    p.avg_pv_measured_upscaled_belgium AS Area_Belgium_pv_production,
    p.avg_pv_measured_upscaled_brussels AS Area_Brussels_pv_production,
    p.avg_pv_measured_upscaled_east_flanders AS Area_EastFlanders_pv_production,
    p.avg_pv_measured_upscaled_flanders AS Area_Flanders_pv_production,
    p.avg_pv_measured_upscaled_flemish_brabant AS Area_FlemishBrabant_pv_production,
    p.avg_pv_measured_upscaled_hainaut AS Area_Hainaut_pv_production,
    p.avg_pv_measured_upscaled_limburg AS Area_Limburg_pv_production,
    p.avg_pv_measured_upscaled_liège AS Area_Liège_pv_production,
    p.avg_pv_measured_upscaled_luxembourg AS Area_Luxembourg_pv_production,
    p.avg_pv_measured_upscaled_namur AS Area_Namur_pv_production,
    p.avg_pv_measured_upscaled_wallonia AS Area_Wallonia_pv_production,
    p.avg_pv_measured_upscaled_walloon_brabant AS Area_WalloonBrabant_pv_production,
    p.avg_pv_measured_upscaled_west_flanders AS Area_WestFlanders_pv_production,
    p.avg_pv_monitored_capacity_antwerp AS Area_Antwerp_pv_capacity,
    p.avg_pv_monitored_capacity_belgium AS Area_Belgium_pv_capacity,
    p.avg_pv_monitored_capacity_brussels AS Area_Brussels_pv_capacity,
    p.avg_pv_monitored_capacity_east_flanders AS Area_EastFlanders_pv_capacity,
    p.avg_pv_monitored_capacity_flanders AS Area_Flanders_pv_capacity,
    p.avg_pv_monitored_capacity_flemish_brabant AS Area_FlemishBrabant_pv_capacity,
    p.avg_pv_monitored_capacity_hainaut AS Area_Hainaut_pv_capacity,
    p.avg_pv_monitored_capacity_limburg AS Area_Limburg_pv_capacity,
    p.avg_pv_monitored_capacity_liège AS Area_Liège_pv_capacity,
    p.avg_pv_monitored_capacity_luxembourg AS Area_Luxembourg_pv_capacity,
    p.avg_pv_monitored_capacity_namur AS Area_Namur_pv_capacity,
    p.avg_pv_monitored_capacity_wallonia AS Area_Wallonia_pv_capacity,
    p.avg_pv_monitored_capacity_walloon_brabant AS Area_WalloonBrabant_pv_capacity,
    p.avg_pv_monitored_capacity_west_flanders AS Area_WestFlanders_pv_capacity,
    w.avg_wind_measured_federal AS Area_Federal_wind_production,
    w.avg_wind_measured_flanders AS Area_Flanders_wind_production,
    w.avg_wind_measured_wallonia AS Area_Wallonia_wind_production,
    w.avg_wind_monitored_capacity_federal AS Area_Federal_wind_capacity,
    w.avg_wind_monitored_capacity_flanders AS Area_Flanders_wind_capacity,
    w.avg_wind_monitored_capacity_wallonia AS Area_Wallonia_wind_capacity,
    t.avg_load,
    i.avg_systemimbalance,
    i.avg_netregulationvolume,
    i.avg_positive_price,
    i.avg_negative_price
FROM pv_production_clean p
LEFT JOIN wind_production_clean w 
    ON p.timestamp = w.timestamp
LEFT JOIN total_load_clean t 
    ON p.timestamp = t.timestamp
LEFT JOIN imbalance_price_clean i 
    ON p.timestamp = i.timestamp;