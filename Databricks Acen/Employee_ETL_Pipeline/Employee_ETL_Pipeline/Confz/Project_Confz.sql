-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

MERGE INTO confz.Project AS T
USING work.Project AS S
  ON T.Proj_ID = S.Proj_ID
     AND T.curr_ind = 'Y'
     AND T.checksum <> S.checksum  
WHEN MATCHED THEN
  UPDATE SET
    T.curr_ind = 'N',
    T.end_date = CURRENT_DATE();

INSERT INTO confz.Project
SELECT
  Proj_Key,  
  S.checksum,
  'Y' AS curr_ind,
  CURRENT_DATE() AS start_date,
  DATE('9999-12-31') AS end_date,
  S.Proj_ID,
  S.Proj_Name,
  S.Start_Date,
  S.End_Date,
  S.Domain,
  S.ProjMang_ID,
  S.ingestTimestamp,
  S.loadKey
FROM work.Project S
LEFT ANTI JOIN confz.Project T
  ON S.checksum = T.checksum;  

-- COMMAND ----------

select * from confz.project;