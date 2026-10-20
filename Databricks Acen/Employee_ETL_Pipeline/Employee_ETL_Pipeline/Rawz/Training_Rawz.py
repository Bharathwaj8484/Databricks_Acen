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

ingest_sequentially_s3(
    table_name="Training",      
    source_bucket=bucket_name,
    source_folder=source_folder_name,
    archive_folder=archive_folder_name,
    file_format="csv",
    access_key=access_key,
    secret_key=secret_key
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from rawz.training;