-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

MERGE INTO confz.Department AS T
USING work.Department AS S
  ON T.Dept_ID = S.Dept_ID
     AND T.curr_ind = 'Y'
     AND T.checksum <> S.checksum  
WHEN MATCHED THEN
  UPDATE SET
    T.curr_ind = 'N',
    T.end_date = CURRENT_DATE();

INSERT INTO confz.Department
SELECT
  Dept_Key,  
  S.checksum,
  'Y' AS curr_ind,
  CURRENT_DATE() AS start_date,
  DATE('9999-12-31') AS end_date,
  S.Dept_ID,
  S.Dept_Name,
  S.Base_Loc,
  S.Dept_MangID,
  S.ingestTimestamp,
  S.loadKey
FROM work.Department S
LEFT ANTI JOIN confz.Department T
  ON S.checksum = T.checksum; 

-- COMMAND ----------

select * from confz.department;

-- COMMAND ----------

select count(*) from confz.department;