-- Databricks notebook source
USE CATALOG hive_metastore;

-- COMMAND ----------

INSERT INTO pubz.result
WITH
  -- 1) Current employees
  cur_emp AS (
    SELECT *
    FROM confz.employee
    WHERE curr_ind = 'Y'
  ),

  -- 2) Current department dimension
  cur_dept AS (
    SELECT Dept_ID, Dept_Name
    FROM confz.Department
    WHERE curr_ind = 'Y'
  ),

  -- 3) Current payroll dimension
  cur_pay AS (
    SELECT Payr_ID, Base_Pay, Bon_Amt, Total_Pay
    FROM confz.Payroll
    WHERE curr_ind = 'Y'
  ),

  -- 4) Current project dimension
  cur_proj AS (
    SELECT Proj_ID, Proj_Name, Domain
    FROM confz.Project
    WHERE curr_ind = 'Y'
  ),

  -- 5) Sum total training hours per employee across history
  emp_train_sum AS (
    SELECT
      e.Emp_ID,
      SUM(t.Duration_Hrs) AS total_training_hours
    FROM confz.employee e
    JOIN confz.Training t
      ON e.Training_ID = t.Training_ID
    GROUP BY e.Emp_ID
  ),

  -- 6) Base result combining everything
  base AS (
    SELECT
      e.Emp_ID,
      e.F_Name,
      e.L_Name,
      e.Email,
      d.Dept_ID,
      d.Dept_Name,
      p.Base_Pay,
      p.Bon_Amt,
      p.Total_Pay,
      pj.Proj_ID,
      pj.Proj_Name,
      pj.Domain,
      COALESCE(ts.total_training_hours, 0) AS total_training_hours
    FROM cur_emp e
    LEFT JOIN cur_dept d
      ON e.Dept_ID = d.Dept_ID
    LEFT JOIN cur_pay p
      ON e.Payroll_ID = p.Payr_ID
    LEFT JOIN cur_proj pj
      ON e.Proj_ID = pj.Proj_ID
    LEFT JOIN emp_train_sum ts
      ON e.Emp_ID = ts.Emp_ID
  ),

  -- 7) Deduplicate, keeping only the highest training total per Emp_ID
  ranked AS (
    SELECT *,
      ROW_NUMBER() OVER (
        PARTITION BY Emp_ID
        ORDER BY total_training_hours DESC
      ) AS rn
    FROM base
  )

SELECT
  Emp_ID,
  F_Name,
  L_Name,
  Email,
  Dept_ID,
  Dept_Name,
  Base_Pay,
  Bon_Amt,
  Total_Pay,
  Proj_ID,
  Proj_Name,
  Domain,
  total_training_hours
FROM ranked
WHERE rn = 1;

-- COMMAND ----------

select * from pubz.result
order by Emp_ID;