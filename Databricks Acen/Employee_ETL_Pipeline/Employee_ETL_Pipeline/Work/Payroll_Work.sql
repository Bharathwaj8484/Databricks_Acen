-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

INSERT INTO work.Payroll
WITH cleaned AS (
  SELECT
    LPAD(TRIM(Payr_ID), 8, '0')                                                    AS Payr_ID,
    CAST(Base_Pay AS DECIMAL(12,2))                                                AS Base_Pay,
    DATE_FORMAT(
      TO_DATE(REPLACE(REPLACE(Pay_Date, '/', '-'), '.', '-'), 'yyyy-MM-dd'),
      'yyyy-MM-dd'
    )                                                                              AS Pay_Date,
    CAST(REGEXP_REPLACE(TRIM(Overtime_Mins), '[^0-9]', '') AS INT)                  AS Overtime_Mins,
    CAST(Deductions AS DECIMAL(12,2))                                               AS Deductions,
    CAST(Incr_Pct AS DECIMAL(5,2))                                                  AS Incr_Pct,
    CAST(Bon_Amt AS DECIMAL(12,2))                                                  AS Bon_Amt,
    CAST(Total_Pay AS DECIMAL(12,2))                                                AS Total_Pay,
    ingestTimestamp                                                              AS ingestTimestamp,
    loadKey                                                                      AS loadKey
  FROM rawz.Payroll
  WHERE Payr_ID IS NOT NULL
),
filtered AS (
  SELECT
    Payr_ID,
    Base_Pay,
    Pay_Date,
    Overtime_Mins,
    Deductions,
    Incr_Pct,
    Bon_Amt,
    Total_Pay,
    ingestTimestamp,
    loadKey,

    -- Compute SHA2(256) checksum over cleaned columns
    SHA2(
      CONCAT_WS('||',
        Payr_ID,
        Base_Pay,
        Pay_Date,
        Overtime_Mins,
        Deductions,
        Incr_Pct,
        Bon_Amt,
        Total_Pay
      ),
      256
    )                                                                              AS checksum
  FROM cleaned
),
ranked AS (
  SELECT
    Payr_ID,
    Base_Pay,
    Pay_Date,
    Overtime_Mins,
    Deductions,
    Incr_Pct,
    Bon_Amt,
    Total_Pay,
    ingestTimestamp,
    loadKey,
    checksum,
    ROW_NUMBER() OVER (
      PARTITION BY checksum
      ORDER BY ingestTimestamp DESC
    )                                                                                AS rn
  FROM filtered
)
SELECT
  uuid() AS Payz_Key,
  Payr_ID,
  Base_Pay,
  Pay_Date,
  Overtime_Mins,
  Deductions,
  Incr_Pct,
  Bon_Amt,
  Total_Pay,
  ingestTimestamp,
  loadKey,
  checksum
FROM ranked
WHERE rn = 1;

-- COMMAND ----------

select count(*) from work.payroll;

-- COMMAND ----------

select * from work.payroll;