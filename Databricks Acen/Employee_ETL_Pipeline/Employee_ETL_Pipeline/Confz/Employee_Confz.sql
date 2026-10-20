-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

MERGE INTO confz.employee AS T
USING work.Employee AS S
  ON T.Emp_ID = S.Emp_ID
     AND T.curr_ind = 'Y'
     AND T.Checksum <> S.checksum
WHEN MATCHED THEN
  UPDATE SET
    T.curr_ind = 'N',
    T.end_date = CURRENT_DATE();

INSERT INTO confz.employee
SELECT
  Emp_Key, 
  S.checksum AS Checksum,
  'Y' AS curr_ind,
  CURRENT_DATE() AS create_date,
  DATE('9999-12-31') AS end_date,
  S.Emp_ID,
  S.F_Name,
  S.L_Name,
  S.Email,
  S.Hire_Date,
  S.Birth_Date,
  S.Ph_Num,
  S.Gender,
  S.Primary_Skill,
  S.Designation,
  S.Training_Status,
  S.Dept_Name,
  S.Dept_ID,
  S.Mang_ID,
  S.Training_ID,
  S.Proj_ID,
  S.Payroll_ID,
  S.ingest_timestamp,
  S.load_key
FROM work.Employee S
LEFT ANTI JOIN confz.employee T
  ON S.checksum = T.Checksum;

-- COMMAND ----------

select * from confz.employee
order by Emp_ID;

-- COMMAND ----------

select count(*) from confz.employee;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC View Current Rows Only

-- COMMAND ----------

SELECT * FROM confz.Employee WHERE curr_ind = 'Y';

-- COMMAND ----------

SELECT * FROM confz.Employee WHERE curr_ind = 'N';

-- COMMAND ----------

select count(*) from confz.employee;

-- COMMAND ----------

select w.Emp_ID,
w.checksum As work_checksum,
c.Checksum As confz_checksum
from work.employee w
Join confz.employee c
on w.Emp_ID = c.Emp_ID
where w.checksum <> c.checksum;

-- COMMAND ----------

CREATE OR REPLACE VIEW confz.emp_view AS
SELECT * FROM confz.employee;

-- COMMAND ----------

select * from confz.emp_view;