# Databricks notebook source
import os

# --- GET CREDENTIALS FROM ENVIRONMENT VARIABLES ---
access_key = os.environ.get("AK")
secret_key = os.environ.get("SK")

# --- VALIDATE ---
if not access_key or not secret_key:
    raise Exception("⚠️ AWS credentials not found in cluster environment variables!")

spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit, col, to_date, coalesce, when
from pyspark.sql.types import *
import re
import random
from datetime import datetime

def ingest_sequentially_s3(table_name, source_bucket, source_folder, archive_folder, file_format, access_key, secret_key):
    """
    Ingests files sequentially from S3.
    Features:
    1. STRICT SCHEMA ALIGNMENT: Prevents column swapping (Phone <-> Gender).
    2. SMART DATE PARSING: Prevents NULL dates by handling MM/dd/yyyy vs yyyy-MM-dd.
    """
    spark.conf.set("fs.s3a.access.key", access_key)
    spark.conf.set("fs.s3a.secret.key", secret_key)
    spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")
  
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    
    source_path = f"s3a://{source_bucket}/{source_folder}/"
    archive_path = f"s3a://{source_bucket}/{archive_folder}/"
    
    print(f"Checking for files in: {source_path}")

    table_prefix_map = {
        "Employee": "emp",
        "Department": "dept",
        "Project": "project",
        "Training": "training",
        "Payroll": "payroll"
    }
    file_prefix = table_prefix_map.get(table_name, table_name.lower()[:3])
    print(f"Looking for files starting with: '{file_prefix}'")

    try:
        files = dbutils.fs.ls(source_path)
    except Exception as e:
        print(f"Error accessing S3 path. Check keys and bucket name.\nError: {e}")
        return

    data_files = []
    date_pattern = re.compile(r"(\d{8})") 

    for fi in files:

        if fi.name.lower().endswith(f".{file_format}") and fi.name.lower().startswith(file_prefix):
            match = date_pattern.search(fi.name)
            if match:
                date_str = match.group(1) 
                try:
                    sort_key = datetime.strptime(date_str, "%m%d%Y").strftime("%Y%m%d")
                    data_files.append({"path": fi.path, "name": fi.name, "sort_key": sort_key})
                except ValueError:
                    print(f"Skipping file with invalid date in filename: {fi.name}")

    sorted_files = sorted(data_files, key=lambda x: x['sort_key'])

    if not sorted_files:
        print(f"No new '{file_prefix}' files found to ingest.")
        return

    for file_info in sorted_files:
        print(f"--------------------------------------------------")
        print(f"Processing file: {file_info['name']}")
        
        try:
            now = datetime.now()
            load_key = f"LCK{now.strftime('%Y%m%d')}{now.strftime('%H%M%S')}{random.randint(1000, 9999)}"

            df = spark.read.format(file_format) \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .load(file_info['path'])

            df = df.withColumn("ingestTimestamp", current_timestamp()) \
                   .withColumn("loadKey", lit(load_key))

            target_table_name = f"rawz.{table_name}"
            target_schema = spark.table(target_table_name).schema

            df_cols_map = {c.lower(): c for c in df.columns}
            
            final_select_cols = []
            
            for field in target_schema:
                target_col_name = field.name
                target_col_lower = target_col_name.lower()
                target_dtype = field.dataType
                
                if target_col_lower in df_cols_map:
                    source_col_name = df_cols_map[target_col_lower]
                    source_col = col(source_col_name)

                    if isinstance(target_dtype, DateType):
                        col_expr = coalesce(
                            to_date(source_col, "MM/dd/yyyy"),
                            to_date(source_col, "M/d/yyyy"),
                            to_date(source_col, "yyyy-MM-dd"),
                            source_col.cast(DateType()) 
                        ).alias(target_col_name)
                    else:
                        col_expr = source_col.cast(target_dtype).alias(target_col_name)
                        
                    final_select_cols.append(col_expr)
                else:
                    final_select_cols.append(lit(None).cast(target_dtype).alias(target_col_name))

            df_ordered = df.select(*final_select_cols)

            df_ordered.write.mode("append").insertInto(target_table_name)
            
            row_count = df_ordered.count()
            print(f"Successfully inserted {row_count} rows into {target_table_name}")

            audit_schema = StructType([
                StructField("file_name", StringType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("load_key", StringType(), True),
                StructField("completion_status", StringType(), True),
                StructField("file_path", StringType(), True),
                StructField("rows_processed", IntegerType(), True)
            ])
            
            audit_df = spark.createDataFrame([
                (file_info['name'], datetime.now(), load_key, "Success", file_info['path'], row_count)
            ], schema=audit_schema)
            
            audit_df.write.format("delta").mode("append").saveAsTable("rawz.audit_log")
            print("Audit log updated.")
            
            destination_path = archive_path + file_info['name']
            dbutils.fs.mv(file_info['path'], destination_path)
            print(f"Moved file to Archive: {destination_path}")

        except Exception as e:
            print(f"FAILED to process file {file_info['name']}: {str(e)}")
            print("Stopping sequential ingestion due to error.")
            break