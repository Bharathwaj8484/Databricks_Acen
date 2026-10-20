# Databricks notebook source
# MAGIC %run "/Workspace/Users/bharathwaj.k.s@accenture.com/Mini_Project_Bharathwaj/DDL & Ingestion/Ingestion_Util"

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG hive_metastore;

# COMMAND ----------

import os

# --- GET CREDENTIALS FROM ENVIRONMENT VARIABLES ---
access_key = os.environ.get("AK")
secret_key = os.environ.get("SK")

# --- VALIDATE ---
if not access_key or not secret_key:
    raise Exception("⚠️ AWS credentials not found in cluster environment variables!")

# --- CONFIGURATION ---
bucket_name = "acen-employee-csv"
source_folder_name = "csv_file"
archive_folder_name = "mini_project/Archive"

# --- RUN INGESTION ---
ingest_sequentially_s3(
    table_name="Employee",
    source_bucket=bucket_name,
    source_folder=source_folder_name,
    archive_folder=archive_folder_name,
    file_format="csv",
    access_key=access_key,
    secret_key=secret_key
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from rawz.employee;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from rawz.employee;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from rawz.employee;

# COMMAND ----------

# %sql

# TRUNCATE TABLE rawz.audit_log;
# TRUNCATE TABLE rawz.employee;
# TRUNCATE TABLE rawz.department;
# TRUNCATE TABLE rawz.project;
# TRUNCATE TABLE rawz.payroll;
# TRUNCATE TABLE rawz.training;

# TRUNCATE TABLE work.employee;
# TRUNCATE TABLE work.department;
# TRUNCATE TABLE work.Project;
# TRUNCATE TABLE work.Training;
# TRUNCATE TABLE work.Payroll;

# TRUNCATE TABLE confz.employee;
# TRUNCATE TABLE confz.Department;
# TRUNCATE TABLE confz.Project;
# TRUNCATE TABLE confz.Training;
# TRUNCATE TABLE confz.Payroll;

# TRUNCATE TABLE pubz.result;

# SELECT 'rawz.audit_log' as TableName, count(*) as Count FROM rawz.audit_log
# UNION ALL SELECT 'rawz.employee', count(*) FROM rawz.employee
# UNION ALL SELECT 'rawz.department', count(*) FROM rawz.department
# UNION ALL SELECT 'rawz.project', count(*) FROM rawz.project
# UNION ALL SELECT 'rawz.payroll', count(*) FROM rawz.payroll
# UNION ALL SELECT 'rawz.training', count(*) FROM rawz.training
# UNION ALL SELECT 'work.employee', count(*) FROM work.employee
# UNION ALL SELECT 'confz.employee', count(*) FROM confz.employee
# UNION ALL SELECT 'pubz.result', count(*) FROM pubz.result;

# COMMAND ----------

import os

# --- GET CREDENTIALS FROM ENVIRONMENT VARIABLES ---
access_key = os.environ.get("AK")
secret_key = os.environ.get("SK")

# --- VALIDATE ---
if not access_key or not secret_key:
    raise Exception("⚠️ AWS credentials not found in cluster environment variables!")

# --- CONFIGURATION ---
bucket_name = "acen-employee-csv"

# --- CONFIGURE SPARK ---
spark.conf.set("fs.s3a.access.key", access_key)
spark.conf.set("fs.s3a.secret.key", secret_key)
spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")

# --- FILE MOVE LOGIC ---
archive_path = f"s3a://{bucket_name}/mini_project/Archive/"
destination_path = f"s3a://{bucket_name}/csv_file/"

print(f"Moving files FROM: {archive_path}")
print(f"Moving files TO:   {destination_path}")

try:
    files = dbutils.fs.ls(archive_path)
    if not files:
        print("No files found in Archive to move.")
    else:
        count = 0
        for file in files:
            if file.size > 0:
                source_file = file.path
                dest_file = destination_path + file.name
                dbutils.fs.mv(source_file, dest_file)
                print(f"Moved: {file.name}")
                count += 1
        print(f"--------------------------------------------------")
        print(f"RESET COMPLETE. Moved {count} files back to source.")
except Exception as e:
    print(f"Error accessing paths: {e}")