from typing import Any, Dict, List, Optional, Literal, Union
from pyspark.sql.functions import (
    explode, split, trim, lower, col, max, count, sum, avg, desc,
    asc, expr, lit, when, regexp_replace, to_date, current_date,
    current_timestamp, coalesce, date_format, row_number
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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)

local_tz = pytz.timezone("Asia/Ho_Chi_Minh")


if __name__ == "__main__":
    data = [
        ["001", "101", "John Doe", "30", "Male", "50000", "2015-01-01"],
        ["002", "101", "Jane Smith", "25", "Female", "45000", "2016-02-15"],
        ["003", "102", "Bob Brown", "35", "Male", "55000", "2014-05-01"],
        ["004", "102", "Alice Lee", "28", "Female", "48000", "2017-09-30"],
        ["005", "103", "Jack Chan", "40", "Male", "60000", "2013-04-01"],
        ["006", "103", "Jill Wong", "32", "Female", "52000", "2018-07-01"],
        ["007", "101", "James Johnson", "42", "Male", "70000", "2012-03-15"],
        ["008", "102", "Kate Kim", "29", "Female", "51000", "2019-10-01"],
        ["009", "103", "Tom Tan", "33", "Male", "58000", "2016-06-01"],
        ["010", "104", "Lisa Lee", "27", "Female", "47000", "2018-08-01"],
        ["011", "104", "David Park", "38", "Male", "65000", "2015-11-01"],
        ["012", "105", "Susan Chen", "31", "Female", "54000", "2017-02-15"],
        ["013", "106", "Brian Kim", "45", "Male", "75000", "2011-07-01"],
        ["014", "107", "Emily Lee", "26", "Female", "46000", "2019-01-01"],
        ["015", "106", "Michael Lee", "37", "Male", "63000", "2014-09-30"],
        ["016", "107", "Kelly Zhang", "30", "Female", "49000", "2018-04-01"],
        ["017", "105", "George Wang", "34", "Male", "57000", "2016-03-15"],
        ["018", "104", "Nancy Liu", "29", "Female", "50000", "2017-06-01"],
        ["019", "103", "Steven Chen", "36", "Male", "62000", "2015-08-01"],
        ["020", "102", "Grace Kim", "32", "Female", "53000", "2018-11-01"]
    ]

    schema = "employee_id string, department_id string, name string, age string, gender string, salary string, hire_date string"
    schema = StructType(
        [
            StructField("employee_id", StringType(), True),
            StructField("department_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("age", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("salary", StringType(), True),
            StructField("hire_date", StringType(), True),
        ]
    )

    spark = (
        SparkSession
        .builder
        .appName("Employees")
        .master("spark://spark-master:7077")
        .getOrCreate()
    )

    df = spark.createDataFrame(data=data, schema=schema, verifySchema=True)
    num_partitions = df.rdd.getNumPartitions()
    logging.info(f"Number of partitions: {num_partitions}.")

    df = df.withColumns(
        {
            "age": col("age").cast(IntegerType()),
            "salary": col("salary").cast(DoubleType())
        }
    )

    df = df.withColumn(
        "gender",
        when(col("gender") == "Male", "M")
        .when(col("gender") == "Female", "F")
        .otherwise(None)
    )
    # df = df.withColumn(
    #     "gender",
    #     expr(
    #         """
    #             case
    #                 when gender = 'Male' then 'M'
    #                 when gender = 'Female' then 'F'
    #             else null
    #             end
    #         """
    #     )
    # )

    df = df.withColumn(
        "hire_date",
        to_date(col("hire_date"), "yyyy-MM-dd")
    )
    df = df.withColumn(
        "uploaded_time",
        lit(datetime.now(tz=local_tz))
    )
    df = df.withColumn("date_now", current_date()).withColumn("timestamp_now", current_timestamp())
    df = df.withColumn(
        "hire_date_str",
        date_format(col("hire_date"), "yyyy/MM/dd")
    )
    df = df.withColumn(
        "time_zone",
        date_format(col("timestamp_now"), "z")
    )

    df = df.withColumn(
        "tax",
        col("salary") * 0.2
    )

    df_filtered = df.select(
        expr("employee_id as emp_id"),
        col("name"),
        expr("cast(age as int) as age"),
        df.salary
    )
    # df_filtered = df.selectExpr(
    #     "employee_id as emp_id",
    #     "name",
    #     "cast(age as int) as age",
    #     "salary"
    # )

    logging.info(f"Cast schema: {df.schema}")

    df_salary_gt_50000 = df.where("salary > 50000")
    logging.info(f"Filtered DataFrame by conditions: salary > 50000.")
    num_partitions = df_salary_gt_50000.rdd.getNumPartitions()
    logging.info(f"Number of partitions: {num_partitions}.")
    path_salary_gt_50000 = Path("datasets/output/employees_salary_gt_50000.csv").as_posix()
    df_salary_gt_50000.write.mode("overwrite").format("csv").save(path_salary_gt_50000)

    df_age_gt_30 = df.select("emp_id", "name", "age", "salary").where("age > 30")
    path_age_gt_30 = Path("datasets/output/employees_age_gt_30.csv").as_posix()
    df_age_gt_30.write.mode("overwrite").format("csv").save(path_age_gt_30)

    df_name = df.withColumn(
        "new_name",
        regexp_replace(col("name"), "J", "Z")
    )
    path_name = Path("datasets/output/employees_name.csv").as_posix()
    df_name.write.mode("overwrite").format("csv").save(path_name)

    df_dated = df.na.drop("all")
    df_dated = df.withColumn(
        "gender",
        coalesce(col("gender")),
        lit("O")
    )

    df_unique = df.distinct()
    df_dept_id = df.select("department_id").distinct()

    df = spark.read.format("csv").load("datasets/input/emp.csv")
# docker exec -it sales-spark-master spark-submit --master spark://172.18.0.2:7077 --deploy-mode client /opt/bitnami/spark/jobs/employees.py
