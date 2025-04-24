from typing import Any, Dict, List, Optional, Literal, Union
from pyspark.sql.functions import (
    explode, split, trim, lower, col, max, count, sum, avg, desc,
    asc, expr, lit, when, regexp_replace, to_date, current_date,
    current_timestamp, coalesce, date_format, row_number, spark_partition_id
)
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType, DateType, TimestampType
from pyspark import SparkConf
from pyspark import SparkContext
import logging
from os.path import abspath
from pathlib import Path
import shutil
from delta import *
# from pyspark.logger import PySparkLogger # version Spark 4.0
from datetime import datetime
import pytz
from pyspark.sql.window import Window


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)

local_tz = pytz.timezone("Asia/Ho_Chi_Minh")


if __name__ == "__main__":
    emp_data_1 = [
        ["001", "101", "John Doe", "30", "Male", "50000", "2015-01-01"],
        ["002", "101", "Jane Smith", "25", "Female", "45000", "2016-02-15"],
        ["003", "102", "Bob Brown", "35", "Male", "55000", "2014-05-01"],
        ["004", "102", "Alice Lee", "28", "Female", "48000", "2017-09-30"],
        ["005", "103", "Jack Chan", "40", "Male", "60000", "2013-04-01"],
        ["006", "103", "Jill Wong", "32", "Female", "52000", "2018-07-01"],
        ["007", "101", "James Johnson", "42", "Male", "70000", "2012-03-15"],
        ["008", "102", "Kate Kim", "29", "Female", "51000", "2019-10-01"],
        ["009", "103", "Tom Tan", "33", "Male", "58000", "2016-06-01"],
        ["010", "104", "Lisa Lee", "27", "Female", "47000", "2018-08-01"]
    ]
    emp_data_2 = [
        ["011", "104", "David Park", "38", "Male", "65000", "2015-11-01"],
        ["012", "105", "Susan Chen", "31", "Female", "54000", "2017-02-15"],
        ["013", "106", "Brian Kim", "45", "Male", "75000", "2011-07-01"],
        ["014", "107", "Emily Lee", "26", "Female", "46000", "2019-01-01"],
        ["015", "106", "Michael Lee", "37", "Male", "63000", "2014-09-30"],
        ["016", "107", "Kelly Zhang", "30", "Female", "49000", "2018-04-01"],
        ["017", "105", "George Wang", "34", "Male", "57000", "2016-03-15"],
        ["018", "104", "Nancy Liu", "29", "", "50000", "2017-06-01"],
        ["019", "103", "Steven Chen", "36", "Male", "62000", "2015-08-01"],
        ["020", "102", "Grace Kim", "32", "Female", "53000", "2018-11-01"]
    ]
    emp_schema = "employee_id string, department_id string, name string, age string, gender string, salary string, hire_date string"

    dept_data = [
        ["101", "Sales", "NYC", "US", "1000000"],
        ["102", "Marketing", "LA", "US", "900000"],
        ["103", "Finance", "London", "UK", "1200000"],
        ["104", "Engineering", "Beijing", "China", "1500000"],
        ["105", "Human Resources", "Tokyo", "Japan", "800000"],
        ["106", "Research and Development", "Perth", "Australia", "1100000"],
        ["107", "Customer Service", "Sydney", "Australia", "950000"]
    ]
    dept_schema = "department_id string, department_name string, city string, country string, budget string"

    spark = (
        SparkSession
        .builder
        .appName("Employees Aggregation")
        .master("spark://sales-spark-master:7077")
        .getOrCreate()
    )

    df_1 = spark.createDataFrame(data=emp_data_1, schema=emp_schema)
    df_2 = spark.createDataFrame(data=emp_data_2, schema=emp_schema)

    df_emp = df_1.unionAll(df_2)
    df_emp = df_emp.orderBy(col("salary").asc())

    df_sum = df_emp.groupBy("department_id").agg(sum("salary").alias("total_dept_salary"))
    path_sum = Path("datasets/output/employees_sum.csv").as_posix()
    df_sum.write.mode("overwrite").format("csv").save(path_sum)

    df_avg = df_emp.groupBy("department_id").agg(avg("salary").alias("avg_dept_salary")).where("avg_dept_salary > 50000")
    path_avg = Path("datasets/output/employees_avg.csv").as_posix()
    df_avg.write.mode("overwrite").format("csv").save(path_avg)

    df_2_other = df_2.select("employee_id", "salary", "department_id", "name", "hire_date", "gender", "age")
    df_emp = df_1.unionByName(df_2_other)

    window_spec = Window.partitionBy(col("department_id")).orderBy(col("salary").desc())
    max_func = max(col("salary")).over(window_spec)

    df_max_salary = df_emp.withColumn("max_salary", max_func)
    path_max_salary = Path("datasets/output/employees_max_salary.csv").as_posix()
    df_max_salary.write.mode("overwrite").format("csv").save(path_max_salary)

    window_spec = Window.partitionBy(col("department_id")).orderBy(col("salary").desc())
    rn = row_number().over(window_spec)

    df_second_salary = df_emp.withColumn("rn", rn).where("rn = 2")
    path_second_salary = Path("datasets/output/employees_second_salary.csv").as_posix()
    df_second_salary.write.mode("overwrite").format("csv").save(path_second_salary)

    df_second_salary_expr = df_emp.withColumn("rn", expr("row_number() over(partition by department_id order by salary desc)")).where("rn = 2")
    path_second_salary_expr = Path("datasets/output/employees_second_salary_expr.csv").as_posix()
    df_second_salary_expr.write.mode("overwrite").format("csv").save(path_second_salary_expr)

    df_dept = spark.createDataFrame(data=dept_data, schema=dept_schema)


    logging.info(f"Number of partitions of df_emp: {df_emp.rdd.getNumPartitions()}.")
    logging.info(f"Number of partitions of df_dept: {df_dept.rdd.getNumPartitions()}.")

    df_emp_partitioned_incre = df_emp.repartition(100)
    logging.info(f"Number of partitions of df_emp_partitioned_incre using repartition: {df_emp_partitioned_incre.rdd.getNumPartitions()}.")
    df_emp_partitioned_decre = df_emp.repartition(4)
    logging.info(f"Number of partitions of df_emp_partitioned_decre using repartition: {df_emp_partitioned_decre.rdd.getNumPartitions()}.")
    df_emp_partitioned_incre = df_emp.coalesce(100)
    logging.info(f"Number of partitions of df_emp_partitioned_incre using coalesce: {df_emp_partitioned_incre.rdd.getNumPartitions()}.")
    df_emp_partitioned_decre = df_emp.coalesce(4)
    logging.info(f"Number of partitions of df_emp_partitioned_decre using coalesce: {df_emp_partitioned_decre.rdd.getNumPartitions()}.")

    df_emp_partitioned = df_emp.repartition(4, "department_id")
    df_1 = df_emp.repartition(4, "department_id").withColumn("partition_num", spark_partition_id())

    df_joined = df_emp.alias("e").join(df_dept.alias("d"), how="inner", on=df_emp.department_id == df_dept.department_id)

    df_joined = df_emp.alias("e").join(df_dept.alias("d"), how="left_outer", on=df_emp.department_id == df_dept.department_id)

    df_joined.select("e.name", "d.department_name", "d.department_id", "e.salary").write.mode("overwrite").format("csv").save("data/output/employees_joined.csv")

    df_final = df_emp.join(
        df_dept,
        how="left_outer",
        on=(
            (df_emp.department_id == df_dept.department_id) &
            ((df_emp.department_id == "101") | (df_emp.department_id == "102")) &
            (df_emp.salary.isNull())
        )
    )

# docker exec -it sales-spark-master spark-submit --master spark://172.18.0.2:7077 --deploy-mode client /opt/bitnami/spark/jobs/employees_agg.py