# Databricks notebook source
# ==============================================================================
# IMPORTS — Readmission Risk Explorer (v2.0)
# Run this cell first in Databricks to install all dependencies.
# ==============================================================================

# COMMAND ----------
%pip install --upgrade langchain langchain-core langchain-databricks plotly gradio markdown
%pip install pandas numpy scikit-learn xgboost
%pip install optuna shap imbalanced-learn
%pip install matplotlib
# COMMAND ----------
# dbutils.library.restartPython()