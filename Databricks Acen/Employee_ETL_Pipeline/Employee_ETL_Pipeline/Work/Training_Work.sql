-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

INSERT INTO work.Training
WITH cleaned AS (
  SELECT
    LPAD(TRIM(Training_ID), 8, '0')                                                   AS Training_ID,

    -- Normalize Training_name by collapsing spaces to underscores and trimming underscores
    TRIM(BOTH '_' FROM REGEXP_REPLACE(
      TRIM(REPLACE(Training_name, ' ', '_')),
      '_+', '_'
    ))                                                                                 AS Training_name,

    -- Normalize Start_Date and End_Date to YYYY-MM-DD
    DATE_FORMAT(
      TO_DATE(REPLACE(REPLACE(Start_Date, '/', '-'), '.', '-'), 'yyyy-MM-dd'),
      'yyyy-MM-dd'
    )                                                                                  AS Start_Date,
    DATE_FORMAT(
      TO_DATE(REPLACE(REPLACE(End_Date, '/', '-'), '.', '-'), 'yyyy-MM-dd'),
      'yyyy-MM-dd'
    )                                                                                  AS End_Date,

    -- Cast Duration_Hrs to integer
    CAST(REGEXP_REPLACE(TRIM(Duration_Hrs), '[^0-9]', '') AS INT)                      AS Duration_Hrs,

    -- Normalize Training_Mode by collapsing spaces to underscores and trimming underscores
    TRIM(BOTH '_' FROM REGEXP_REPLACE(
      TRIM(REPLACE(Training_Mode, ' ', '_')),
      '_+', '_'
    ))                                                                                 AS Training_Mode,

    ingestTimestamp                                                               AS ingestTimestamp,
    loadKey                                                                       AS loadKey
  FROM rawz.Training
  WHERE Training_ID IS NOT NULL
),
filtered AS (
  SELECT
    Training_ID,
    Training_name,
    Start_Date,
    End_Date,
    Duration_Hrs,
    Training_Mode,
    ingestTimestamp,
    loadKey,

    -- Compute SHA2(256) checksum over cleaned columns
    SHA2(
      CONCAT_WS('||',
        Training_name,
        Start_Date,
        End_Date,
        Duration_Hrs,
        Training_Mode
      ),
      256
    )                                                                               AS checksum
  FROM cleaned
),
ranked AS (
  SELECT
    Training_ID,
    Training_name,
    Start_Date,
    End_Date,
    Duration_Hrs,
    Training_Mode,
    ingestTimestamp,
    loadKey,
    checksum,
    ROW_NUMBER() OVER (
      PARTITION BY checksum
      ORDER BY ingestTimestamp DESC
    )                                                                                 AS rn
  FROM filtered
)
SELECT
  uuid() AS Training_Key,
  Training_ID,
  Training_name,
  Start_Date,
  End_Date,
  Duration_Hrs,
  Training_Mode,
  ingestTimestamp,
  loadKey,
  checksum
FROM ranked
WHERE rn = 1;

-- COMMAND ----------

select count(*) from work.Training;

-- COMMAND ----------

select * from work.training;

-- COMMAND ----------

-- insert into work.Training(
--   Training_Key,
--   Training_ID,
--   Training_name,
--   Start_Date,
--   End_Date,
--   Duration_Hrs,
--   Training_Mode,
--   ingestTimestamp,
--   loadKey,
--   checksum
-- )
-- VALUES (
--   '99ae0392-850c-49e4-bd48-6dbe481b0a7719ae0392-850c-49e4-df48-6dbe119b0a77',
--   '000T0001',
--   'Python for Data Science',
--   '2022-01-01',
--   '2022-07-09',   
--   100,
--   'Online',
--   current_timestamp(),
--   'LCK202506081431185088',
--   sha2(concat('Python for Data Science','2022-01-01','2022-07-09',100,'Online'),256)
-- );