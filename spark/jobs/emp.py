from typing import Any, Dict, List, Optional, Literal, Union
from pyspark.sql.functions import (
    explode, split, trim, lower, col, max, count, sum, avg, desc,
    asc, expr, lit, when, regexp_replace, to_date, current_date,
    current_timestamp, coalesce, date_format, row_number, from_json,
    to_json, spark_partition_id
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
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)

local_tz = pytz.timezone("Asia/Ho_Chi_Minh")


def get_time(func):
    def inner_get_time() -> str:
        start_time = time.time()
        func()
        end_time = time.time()
        return (f"Execution time: {(end_time - start_time) * 1000} ms")

    print(inner_get_time())


if __name__ == "__main__":
    spark = (
        SparkSession
        .builder
        .appName("Employees")
        .master("spark://sales-spark-master:7077")
        .getOrCreate()
    )

    df = (
        spark
        .read
        .format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load("datasets/input/emp.csv")
    )

    _schema = "employee_id int, department_id int, name string, age int, gender string, salary double, hire_date date"
    df_schema = (spark.read
                 .format("csv")
                 .option("header", True)
                 .schema(_schema)
                 .load("datasets/input/emp.csv"))

    _schema = "employee_id int, department_id int, name string, age int, gender string, salary double, hire_date date, bad_record string"
    df_p = (spark.read
            .format("csv")
            .schema(_schema)
            .option("columnNameOfCorruptRecord", "bad_record")
            .option("header",True)
            .load("datasets/input/emp_new.csv"))
    path_p = Path("datasets/output/employees_permissive.csv").as_posix()
    df_p.write.mode("overwrite").format("csv").save(path_p)

    _schema = "employee_id int, department_id int, name string, age int, gender string, salary double, hire_date date"
    df_m = (spark.read
            .format("csv")
            .option("header", True)
            .option("mode", "DROPMALFORMED")
            .schema(_schema)
            .load("datasets/input/emp_new.csv"))
    path_m = Path("datasets/output/employees_dropmalformed.csv").as_posix()
    df_m.write.mode("overwrite").format("csv").save(path_m)

    _schema = "employee_id int, department_id int, name string, age int, gender string, salary double, hire_date date"
    df_f = (spark.read
            .format("csv")
            .option("header", True)
            .option("mode", "FAILFAST")
            .schema(_schema)
            .load("datasets/input/emp_new.csv"))
    path_f = Path("datasets/output/employees_failfast.csv").as_posix()
    df_f.write.mode("overwrite").format("csv").save(path_f)

    _options = {
        "header": "true",
        "inferSchema": "true",
        "mode": "PERMISSIVE"
    }
    df = (spark.read.format("csv").options(**_options).load("datasets/input/emp.csv"))

    df_parquet = spark.read.format("parquet").load("datasets/input/sales_total_parquet/*.parquet")

    df_orc = spark.read.format("orc").load("datasets/input/sales_total_orc/*.orc")


    @get_time
    def x():
        df = spark.read.format("parquet").load("datasets/input/sales_data.parquet")
        df.count()


    @get_time
    def x():
        df = spark.read.format("parquet").load("datasets/input/sales_data.parquet")
        df.select("trx_id").count()


    df_1 = spark.read.format("parquet").load("datasets/input/sales_recursive/sales_1/1.parquet")

    df_2 = spark.read.format("parquet").load("datasets/input/sales_recursive/sales_1/sales_2/2.parquet")

    df_r = spark.read.format("parquet").option("recursiveFileLookup", True).load("data/input/sales_recursive/")
    path_r = Path("datasets/output/employees_recursive.csv").as_posix()
    df_r.write.mode("overwrite").format("csv").save(path_r)

    df_single = spark.read.format("json").load("datasets/input/order_singleline.json")

    df_multi = spark.read.format("json").option("multiLine", True).load("datasets/input/order_multiline.json")

    _schema = "customer_id string, order_id string, contact array<long>"
    df_schema = spark.read.format("json").schema(_schema).load("data/input/order_singleline.json")

    _schema = "contact array<string>, customer_id string, order_id string, order_line_items array<struct<amount double, item_id string, qty long>>"
    df_schema_new = spark.read.format("json").schema(_schema).load("data/input/order_singleline.json")

    df_expanded = df.withColumn("parsed", from_json(df.value, _schema))

    df_unparsed = df_expanded.withColumn("unparsed", to_json(df_expanded.parsed))

    df_1 = df_expanded.select("parsed.*")

    df_2 = df_1.withColumn("expanded_line_items", explode("order_line_items"))

    df_3 = df_2.select("contact", "customer_id", "order_id", "expanded_line_items.*")

    df_final = df_3.withColumn("contact_expanded", explode("contact"))
    path_final = Path("datasets/output/employees_recursive.csv").as_posix()
    df_final.write.mode("overwrite").format("csv").save(path_final)