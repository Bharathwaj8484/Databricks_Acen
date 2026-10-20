-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

-- A-B Testing: Rows in work but not in confz (by checksum)
SELECT 'Employee' AS TableName, COUNT(*) AS MissingInConfz
FROM work.Employee w
LEFT ANTI JOIN confz.Employee c ON w.checksum = c.checksum

UNION ALL

SELECT 'Department', COUNT(*)
FROM work.Department w
LEFT ANTI JOIN confz.Department c ON w.checksum = c.checksum

UNION ALL

SELECT 'Project', COUNT(*)
FROM work.Project w
LEFT ANTI JOIN confz.Project c ON w.checksum = c.checksum

UNION ALL

SELECT 'Payroll', COUNT(*)
FROM work.Payroll w
LEFT ANTI JOIN confz.Payroll c ON w.checksum = c.checksum

UNION ALL

SELECT 'Training', COUNT(*)
FROM work.Training w
LEFT ANTI JOIN confz.Training c ON w.checksum = c.checksum;


-- COMMAND ----------

-- Row Count Comparison: Total rows in work vs confz
SELECT 'Employee' AS TableName,
       (SELECT COUNT(*) FROM work.Employee) AS work_count,
       (SELECT COUNT(*) FROM confz.Employee) AS confz_count

UNION ALL

SELECT 'Department',
       (SELECT COUNT(*) FROM work.Department),
       (SELECT COUNT(*) FROM confz.Department)

UNION ALL

SELECT 'Project',
       (SELECT COUNT(*) FROM work.Project),
       (SELECT COUNT(*) FROM confz.Project)

UNION ALL

SELECT 'Payroll',
       (SELECT COUNT(*) FROM work.Payroll),
       (SELECT COUNT(*) FROM confz.Payroll)

UNION ALL

SELECT 'Training',
       (SELECT COUNT(*) FROM work.Training),
       (SELECT COUNT(*) FROM confz.Training);
