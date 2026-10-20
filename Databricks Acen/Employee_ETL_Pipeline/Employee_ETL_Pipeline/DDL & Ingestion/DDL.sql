-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

-- DBTITLE 1,Cell 1
-- DROP TABLE IF EXISTS rawz.employee;
-- DROP TABLE IF EXISTS rawz.department;
-- DROP TABLE IF EXISTS rawz.training;
-- DROP TABLE IF EXISTS rawz.project;
-- DROP TABLE IF EXISTS rawz.payroll;
-- DROP TABLE IF EXISTS work.employee;
-- DROP TABLE IF EXISTS work.department;
-- DROP TABLE IF EXISTS work.project;
-- DROP TABLE IF EXISTS work.training;
-- DROP TABLE IF EXISTS work.payroll;
-- DROP TABLE IF EXISTS confz.employee;
-- DROP TABLE IF EXISTS confz.department;
-- DROP TABLE IF EXISTS confz.project;
-- DROP TABLE IF EXISTS confz.payroll;
-- DROP TABLE IF EXISTS confz.training;
-- DROP TABLE IF EXISTS pubz.result;
-- DROP TABLE IF EXISTS rawz.audit_log

-- COMMAND ----------

-- ----COMMAND ----------
-- -- 1. Clean up RAWZ Zone (Ingestion & Audit)
-- TRUNCATE TABLE rawz.audit_log;
-- TRUNCATE TABLE rawz.employee;
-- TRUNCATE TABLE rawz.department;
-- TRUNCATE TABLE rawz.project;
-- TRUNCATE TABLE rawz.payroll;
-- TRUNCATE TABLE rawz.training;

-- -- 2. Clean up WORK Zone (Transformed Data)
-- TRUNCATE TABLE work.employee;
-- TRUNCATE TABLE work.department;
-- TRUNCATE TABLE work.Project;
-- TRUNCATE TABLE work.Training;
-- TRUNCATE TABLE work.Payroll;

-- -- 3. Clean up CONFZ Zone (Conformed Data)
-- TRUNCATE TABLE confz.employee;
-- TRUNCATE TABLE confz.Department;
-- TRUNCATE TABLE confz.Project;
-- TRUNCATE TABLE confz.Training;
-- TRUNCATE TABLE confz.Payroll;

-- -- 4. Clean up PUBZ Zone (Publication/Results)
-- TRUNCATE TABLE pubz.result;

-- -- COMMAND ----------
-- -- 5. Verification: Check that counts are 0
-- SELECT 'rawz.audit_log' as TableName, count(*) as Count FROM rawz.audit_log
-- UNION ALL SELECT 'rawz.employee', count(*) FROM rawz.employee
-- UNION ALL SELECT 'rawz.department', count(*) FROM rawz.department
-- UNION ALL SELECT 'rawz.project', count(*) FROM rawz.project
-- UNION ALL SELECT 'rawz.payroll', count(*) FROM rawz.payroll
-- UNION ALL SELECT 'rawz.training', count(*) FROM rawz.training
-- UNION ALL SELECT 'work.employee', count(*) FROM work.employee
-- UNION ALL SELECT 'confz.employee', count(*) FROM confz.employee
-- UNION ALL SELECT 'pubz.result', count(*) FROM pubz.result;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Audit Table

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS rawz;

CREATE TABLE IF NOT EXISTS rawz.audit_log (
    file_name STRING,
    timestamp TIMESTAMP,
    load_key STRING,
    completion_status STRING,
    file_path STRING,
    rows_processed INT
)
USING DELTA;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # RAWZ

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS rawz;
USE rawz;

CREATE TABLE IF NOT EXISTS rawz.employee (
  Emp_ID           VARCHAR(50),
  F_Name           VARCHAR(50),
  L_Name           VARCHAR(50),
  Email            VARCHAR(50),
  Hire_Date        DATE,
  Birth_Date       DATE,
  Ph_Num           VARCHAR(50),
  Gender           VARCHAR(50),
  Primary_Skill    VARCHAR(50),
  Designation      VARCHAR(50),
  Training_Status  VARCHAR(50),
  Dept_Name        VARCHAR(50),
  Dept_ID          VARCHAR(50),
  Mang_ID          VARCHAR(50),
  Training_ID      VARCHAR(50),
  Proj_ID          VARCHAR(50),
  Payroll_ID       VARCHAR(50),
  ingestTimestamp TIMESTAMP,
  loadKey         VARCHAR(50)
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS rawz;
USE rawz;

CREATE TABLE IF NOT EXISTS rawz.department (
  Dept_ID          VARCHAR(50),
  Dept_Name        VARCHAR(50),
  Base_Loc         VARCHAR(50),
  Dept_MangID      VARCHAR (50),
  ingestTimestamp  TIMESTAMP,
  loadKey          VARCHAR(50)
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS rawz;
USE rawz;

CREATE TABLE IF NOT EXISTS rawz.project (
  Proj_ID          VARCHAR(50),
  Proj_Name        VARCHAR(50),
  Start_Date       DATE,
  End_Date         DATE,
  ProjMang_ID      VARCHAR(50),
  Domain           VARCHAR(50),
  ingestTimestamp  TIMESTAMP,
  loadKey          VARCHAR(50)
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS rawz;
USE rawz;

CREATE TABLE IF NOT EXISTS rawz.training (
  Training_ID      VARCHAR(50),
  Training_name    VARCHAR(50),
  Start_Date       DATE,
  End_Date         DATE,
  Duration_Hrs     INT,
  Training_Mode    VARCHAR(50),
  ingestTimestamp  TIMESTAMP,
  loadKey          VARCHAR(50)
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS rawz;
USE rawz;

CREATE TABLE IF NOT EXISTS rawz.payroll (
  Payr_ID          VARCHAR(50),
  Base_Pay         DECIMAL(12,2),
  Pay_Date         DATE,
  Overtime_Mins    DECIMAL(12,2),
  Deductions       DECIMAL(12,2),
  Incr_Pct         DECIMAL(12,2),
  Bon_Amt          DECIMAL(12,2),
  Total_Pay        DECIMAL(12,2),
  ingestTimestamp  TIMESTAMP,
  loadKey          VARCHAR(50)
)
USING DELTA;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # WORK

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS work;
USE work;

DROP TABLE IF EXISTS work.employee;

CREATE TABLE work.employee (
  Emp_Key          STRING,
  Emp_ID           STRING,
  F_Name           STRING,
  L_Name           STRING,
  Email            STRING,
  Hire_Date        DATE,
  Birth_Date       DATE,
  Ph_Num           STRING,
  Gender           STRING,
  Primary_Skill    STRING,
  Designation      STRING,
  Training_Status  STRING,
  Dept_Name        STRING,
  Dept_ID          STRING,
  Mang_ID          STRING,
  Training_ID      STRING,
  Proj_ID          STRING,
  Payroll_ID       STRING,
  ingest_timestamp TIMESTAMP,
  load_key         STRING,
  checksum         STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS work;
USE work;

DROP TABLE IF EXISTS work.department;

CREATE TABLE work.department (
  Dept_Key          STRING,    
  Dept_ID           STRING,
  Dept_Name         STRING,
  Base_Loc          STRING,
  Dept_MangID       STRING,
  ingestTimestamp   TIMESTAMP,
  loadKey           STRING,
  checksum          STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS work;
USE work;

DROP TABLE IF EXISTS work.Project;

CREATE TABLE work.Project (
  Proj_Key        STRING,
  Proj_ID         STRING,
  Proj_Name       STRING,
  Start_Date      DATE,
  End_Date        DATE,
  Domain          STRING,
  ProjMang_ID     STRING,
  ingestTimestamp TIMESTAMP,
  loadKey         STRING,
  checksum        STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS work;
USE work;

DROP TABLE IF EXISTS work.Training;

CREATE TABLE work.Training (
  Training_Key     STRING,
  Training_ID      STRING,
  Training_name    STRING,
  Start_Date       DATE,
  End_Date         DATE,
  Duration_Hrs     INT,
  Training_Mode    STRING,
  ingestTimestamp  TIMESTAMP,
  loadKey          STRING,
  checksum         STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS work;
USE work;

DROP TABLE IF EXISTS work.Payroll;

CREATE TABLE work.Payroll (
  Payr_Key         STRING,
  Payr_ID          STRING,
  Base_Pay         DECIMAL(12,2),
  Pay_Date         DATE,
  Overtime_Mins    INT,
  Deductions       DECIMAL(12,2),
  Incr_Pct         DECIMAL(5,2),
  Bon_Amt          DECIMAL(12,2),
  Total_Pay        DECIMAL(12,2),
  ingestTimestamp  TIMESTAMP,
  loadKey          STRING,
  checksum         STRING
)
USING DELTA;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # CONFZ ZONE

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS confz;
USE confz;

CREATE TABLE IF NOT EXISTS confz.employee (
  Emp_Key             STRING,        
  Checksum            STRING,        
  curr_ind            STRING,       
  create_date         DATE,          
  end_date            DATE,        
  Emp_ID              STRING,        
  F_Name              STRING,
  L_Name              STRING,
  Email               STRING,
  Hire_Date           DATE,
  Birth_Date          DATE,
  Ph_Num              STRING,
  Gender              STRING,
  Primary_Skill       STRING,
  Designation         STRING,
  Training_Status     STRING,
  Dept_Name           STRING,
  Dept_ID             STRING,
  Mang_ID             STRING,
  Training_ID         STRING,
  Proj_ID             STRING,
  Payroll_ID          STRING,
  ingest_timestamp    TIMESTAMP,
  load_key            STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS confz;
USE confz;

CREATE TABLE IF NOT EXISTS confz.Department (
  Dept_Key         STRING,   
  checksum         STRING,   
  curr_ind         STRING,    
  create_date       DATE,     
  end_date         DATE,     
  Dept_ID          STRING,    
  Dept_Name        STRING,
  Base_Loc         STRING,
  Dept_MangID      STRING,
  ingest_timestamp TIMESTAMP,
  load_key         STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS confz;
USE confz;

CREATE TABLE IF NOT EXISTS confz.Payroll (
  Payr_Key         STRING,   
  checksum         STRING,  
  curr_ind         STRING,  
  create_date       DATE,     
  end_date         DATE,      
  Payr_ID          STRING,   
  Base_Pay         DECIMAL(12,2),
  Pay_Date         DATE,
  Overtime_Mins    INT,
  Deductions       DECIMAL(12,2),
  Incr_Pct         DECIMAL(5,2),
  Bon_Amt          DECIMAL(12,2),
  Total_Pay        DECIMAL(12,2),
  ingest_timestamp TIMESTAMP,
  load_key         STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS confz;
CREATE TABLE IF NOT EXISTS confz.Project (
  Proj_Key         STRING,   
  checksum         STRING,    
  curr_ind         STRING,   
  create_date       DATE,     
  data_end_date         DATE,      
  Proj_ID          STRING, 
  Proj_Name        STRING,
  Start_Date       DATE,
  End_Date         DATE,
  Domain           STRING,
  ProjMang_ID      STRING,
  ingest_timestamp TIMESTAMP,
  load_key         STRING
)
USING DELTA;

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS confz;
USE confz;

CREATE TABLE IF NOT EXISTS confz.Training (
  Trng_Key         STRING,    
  checksum         STRING,    
  curr_ind         STRING,    
  create_date       DATE,     
  data_end_date       DATE,     
  Training_ID      STRING,    
  Training_name    STRING,
  Start_Date       DATE,
  End_Date         DATE,
  Duration_Hrs     INT,
  Training_Mode    STRING,
  ingest_timestamp TIMESTAMP,
  load_key         STRING
)
USING DELTA;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # PUBZ

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS pubz;
USE pubz;

DROP TABLE IF EXISTS pubz.result;

CREATE TABLE pubz.result (
  Emp_ID               STRING,
  F_Name               STRING,
  L_Name               STRING,
  Email                STRING,
  Dept_ID              STRING,
  Dept_Name            STRING,
  Base_Pay             DECIMAL(12,2),
  Bon_Amt              DECIMAL(12,2),
  Total_Pay            DECIMAL(12,2),
  Proj_ID              STRING,
  Proj_Name            STRING,
  Domain               STRING,
  total_training_hours INT
)
USING DELTA;