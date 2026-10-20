# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG hive_metastore;

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO work.department
# MAGIC WITH cleaned AS (
# MAGIC   SELECT
# MAGIC     LPAD(TRIM(Dept_ID), 8, '0')                                                     AS Dept_ID,
# MAGIC
# MAGIC     TRIM(BOTH '_' FROM REGEXP_REPLACE(
# MAGIC       TRIM(REPLACE(Dept_Name, ' ', '_')),
# MAGIC       '_+', '_'
# MAGIC     ))                                                                              AS Dept_Name,
# MAGIC
# MAGIC     TRIM(BOTH '_' FROM REGEXP_REPLACE(
# MAGIC       TRIM(REPLACE(Base_Loc, ' ', '_')),
# MAGIC       '_+', '_'
# MAGIC     ))                                                                              AS Base_Loc,
# MAGIC
# MAGIC     CASE 
# MAGIC       WHEN Dept_MangID IS NOT NULL THEN LPAD(TRIM(Dept_MangID), 8, '0')
# MAGIC       ELSE NULL
# MAGIC     END                                                                              AS Dept_MangID,
# MAGIC
# MAGIC     ingestTimestamp                                                               AS ingestTimestamp,
# MAGIC     loadKey                                                                       AS loadKey
# MAGIC   FROM rawz.Department
# MAGIC   WHERE Dept_ID IS NOT NULL
# MAGIC ),
# MAGIC filtered AS (
# MAGIC   SELECT
# MAGIC     Dept_ID,
# MAGIC     Dept_Name,
# MAGIC     Base_Loc,
# MAGIC     Dept_MangID,
# MAGIC     ingestTimestamp,
# MAGIC     loadKey,
# MAGIC
# MAGIC     SHA2(
# MAGIC       CONCAT_WS('||',
# MAGIC         Dept_ID,
# MAGIC         Dept_Name,
# MAGIC         Base_Loc,
# MAGIC         COALESCE(Dept_MangID, '')
# MAGIC       ),
# MAGIC       256
# MAGIC     )                                                                               AS checksum
# MAGIC   FROM cleaned
# MAGIC ),
# MAGIC ranked AS (
# MAGIC   SELECT
# MAGIC     Dept_ID,
# MAGIC     Dept_Name,
# MAGIC     Base_Loc,
# MAGIC     Dept_MangID,
# MAGIC     ingestTimestamp,
# MAGIC     loadKey,
# MAGIC     checksum,
# MAGIC     ROW_NUMBER() OVER (
# MAGIC       PARTITION BY checksum
# MAGIC       ORDER BY ingestTimestamp DESC
# MAGIC     )                                                                                 AS rn
# MAGIC   FROM filtered
# MAGIC )
# MAGIC SELECT
# MAGIC   uuid() AS Dept_Key,
# MAGIC   Dept_ID,
# MAGIC   Dept_Name,
# MAGIC   Base_Loc,
# MAGIC   Dept_MangID,
# MAGIC   ingestTimestamp,
# MAGIC   loadKey,
# MAGIC   checksum
# MAGIC FROM ranked
# MAGIC WHERE rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM work.Department;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from work.department;