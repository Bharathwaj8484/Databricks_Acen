-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

INSERT INTO work.Project
WITH cleaned AS (
  SELECT
    LPAD(TRIM(Proj_ID), 8, '0')                                                     AS Proj_ID,

    TRIM(BOTH '_' FROM REGEXP_REPLACE(
      TRIM(REPLACE(Proj_Name, ' ', '_')),
      '_+', '_'
    ))                                                                                AS Proj_Name,

    -- Normalize Start_Date and End_Date to YYYY-MM-DD
    DATE_FORMAT(
      TO_DATE(REPLACE(REPLACE(Start_Date, '/', '-'), '.', '-'), 'yyyy-MM-dd'),
      'yyyy-MM-dd'
    )                                                                                 AS Start_Date,
    DATE_FORMAT(
      TO_DATE(REPLACE(REPLACE(End_Date, '/', '-'), '.', '-'), 'yyyy-MM-dd'),
      'yyyy-MM-dd'
    )                                                                                 AS End_Date,

    -- Normalize Domain by collapsing spaces to underscores and trimming underscores
    TRIM(BOTH '_' FROM REGEXP_REPLACE(
      TRIM(REPLACE(Domain, ' ', '_')),
      '_+', '_'
    ))                                                                                AS Domain,

    -- Pad ProjMang_ID to 8 characters if present
    CASE
      WHEN ProjMang_ID IS NOT NULL THEN LPAD(TRIM(ProjMang_ID), 8, '0')
      ELSE NULL
    END                                                                                AS ProjMang_ID,

    ingestTimestamp                                                               AS ingestTimestamp,
    loadKey                                                                       AS loadKey
  FROM rawz.Project
  WHERE Proj_ID IS NOT NULL
),
filtered AS (
  SELECT
    Proj_ID,
    Proj_Name,
    Start_Date,
    End_Date,
    Domain,
    ProjMang_ID,
    ingestTimestamp,
    loadKey,

    -- Compute SHA2(256) checksum over cleaned fields
    SHA2(
      CONCAT_WS('||',
        Proj_Name,
        Start_Date,
        End_Date,
        Domain,
        COALESCE(ProjMang_ID, '')
      ),
      256
    ) AS checksum
  FROM cleaned
),
ranked AS (
  SELECT
    Proj_ID,
    Proj_Name,
    Start_Date,
    End_Date,
    Domain,
    ProjMang_ID,
    ingestTimestamp,
    loadKey,
    checksum,
    ROW_NUMBER() OVER (
      PARTITION BY checksum
      ORDER BY ingestTimestamp DESC
    ) AS rn
  FROM filtered
)
SELECT
  uuid() AS Proj_Key,
  Proj_ID,
  Proj_Name,
  Start_Date,
  End_Date,
  Domain,
  ProjMang_ID,
  ingestTimestamp,
  loadKey,
  checksum
FROM ranked
WHERE rn = 1;

-- COMMAND ----------

select count(*) from work.Project;