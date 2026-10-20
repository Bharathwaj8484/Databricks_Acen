-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------


SET spark.sql.legacy.timeParserPolicy = LEGACY;

INSERT INTO work.employee

WITH cleaned AS (
    SELECT
        LPAD(TRIM(Emp_ID), 8, '0')                        AS Emp_ID,
        REGEXP_REPLACE(TRIM(F_Name), '\\s+', '_')         AS F_Name,
        REGEXP_REPLACE(TRIM(L_Name), '\\s+', '_')         AS L_Name,
        LOWER(TRIM(Email))                                AS Email,

        CAST(Hire_Date AS DATE)                           AS Hire_Date,
        CAST(Birth_Date AS DATE)                          AS Birth_Date,
        
        REGEXP_REPLACE(Ph_Num, '[^0-9]', '')              AS Ph_Num,
        TRIM(Gender)                                      AS Gender,
        REGEXP_REPLACE(TRIM(Primary_Skill), '\\s+', '_')  AS Primary_Skill,
        TRIM(Designation)                                 AS Designation,
        TRIM(Training_Status)                             AS Training_Status,
        REGEXP_REPLACE(TRIM(Dept_Name), '\\s+', '_')      AS Dept_Name,
        LPAD(TRIM(Dept_ID), 8, '0')                       AS Dept_ID,

        CASE 
            WHEN Mang_ID IS NOT NULL AND UPPER(TRIM(Mang_ID)) NOT IN ('NULL', '')
            THEN LPAD(TRIM(Mang_ID), 8, '0') 
            ELSE NULL 
        END                                               AS Mang_ID,
        
        LPAD(TRIM(Training_ID), 8, '0')                   AS Training_ID,
        LPAD(TRIM(Proj_ID), 8, '0')                       AS Proj_ID,
        LPAD(TRIM(Payroll_ID), 8, '0')                    AS Payroll_ID,
        
        ingestTimestamp,
        loadKey
    FROM rawz.employee
),
filtered AS (
    SELECT *,
        CASE
            WHEN Email NOT RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$' THEN 'Invalid Email'
            WHEN LENGTH(Ph_Num) < 7                                                 THEN 'Invalid Phone'
            WHEN Birth_Date IS NOT NULL AND Hire_Date IS NOT NULL 
                 AND Birth_Date >= Hire_Date                                        THEN 'Invalid Date Range'
            ELSE NULL
        END AS filter_reason,

        SHA2(
            CONCAT_WS('||', 
                Emp_ID, F_Name, L_Name, Email, Hire_Date, Birth_Date, 
                Ph_Num, Gender, Primary_Skill, Designation, Training_Status,
                Dept_Name, Dept_ID, COALESCE(Mang_ID, ''), 
                Training_ID, Proj_ID, Payroll_ID
            ), 256
        ) AS checksum
    FROM cleaned
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY checksum ORDER BY ingestTimestamp DESC) AS rn
    FROM filtered
)
SELECT 
    uuid() AS Emp_Key,
    Emp_ID,
    F_Name,
    L_Name,
    Email,
    Hire_Date,
    Birth_Date,
    Ph_Num,
    Gender,
    Primary_Skill,
    Designation,
    Training_Status,
    Dept_Name,
    Dept_ID,
    Mang_ID,
    Training_ID,
    Proj_ID,
    Payroll_ID,
    ingestTimestamp,
    loadKey,
    checksum
FROM ranked
WHERE filter_reason IS NULL 
  AND rn = 1;

SELECT count(*) as Valid_Rows FROM work.employee;

-- COMMAND ----------

select * from work.Employee;

-- COMMAND ----------

select count(*) from work.employee;

-- COMMAND ----------

-- Sample insertion for testing purpose
-- INSERT INTO work.Employee (
--   Emp_ID,
--   F_Name,
--   L_Name,
--   Email,
--   Hire_Date,
--   Birth_Date,
--   Ph_Num,
--   Gender,
--   Primary_Skill,
--   Designation,
--   Training_Status,
--   Dept_Name,
--   Dept_ID,
--   Mang_ID,
--   Training_ID,
--   Proj_ID,
--   Payroll_ID,
--   ingest_timestamp,
--   load_key,
--   checksum
-- )
-- VALUES (
--   '000E0001',
--   'Nathaniel',
--   'Taylor',
--   'jacobsuarez@example.net',
--   DATE '2010-09-19',
--   DATE '1986-05-17',
--   '1235520394',
--   'M',
--   'Programming_-_Python',
--   'Senior Manager',
--   'Completed',
--   'Software_Development',
--   '00000D01',
--   '0000NULL',
--   '000T0001',
--   '000P0001',
--   '00PR0001',
--   TIMESTAMP '2025-06-03T08:16:37.958',
--   'LCK20250603081637542',
--   '0a46c35e38f16d76cb5d0523e481c52e174808881a957bcbc3e543228766667'
-- );