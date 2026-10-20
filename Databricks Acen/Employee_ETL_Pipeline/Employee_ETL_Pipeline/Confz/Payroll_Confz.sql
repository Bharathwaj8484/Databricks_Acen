-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

MERGE INTO confz.Payroll AS T
USING work.Payroll AS S
  ON T.Payr_ID = S.Payr_ID
     AND T.curr_ind = 'Y'
     AND T.checksum <> S.checksum  
WHEN MATCHED THEN
  UPDATE SET
    T.curr_ind = 'N',
    T.end_date = CURRENT_DATE();

INSERT INTO confz.Payroll
SELECT
  Payr_Key, 
  S.checksum,
  'Y' AS curr_ind,
  CURRENT_DATE() AS start_date,
  DATE('9999-12-31') AS end_date,
  S.Payr_ID,
  S.Base_Pay,
  S.Pay_Date,
  S.Overtime_Mins,
  S.Deductions,
  S.Incr_Pct,
  S.Bon_Amt,
  S.Total_Pay,
  S.ingestTimestamp,
  S.loadKey
FROM work.Payroll S
LEFT ANTI JOIN confz.Payroll T
  ON S.checksum = T.checksum; 

-- COMMAND ----------

select * from confz.payroll;