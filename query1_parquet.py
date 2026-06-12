import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit, udf
from pyspark.sql.types import StringType


spark = SparkSession.builder \
    .appName("LA_Crime_Query1_Parquet") \
    .getOrCreate()

# Parquet data
path_1 = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/user/dsml00292/LA_Crime_Data_Parquet/LA_Crime_Data_2010_2019.parquet"
path_2 = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/user/dsml00292/LA_Crime_Data_Parquet/LA_Crime_Data_2020_2025.parquet"

df_1 = spark.read.parquet(path_1)
df_2 = spark.read.parquet(path_2)
df_all = df_1.union(df_2)

time_col = "TIME OCC"
premise_col = "Premis Desc"


df_cleaned = df_all.withColumn("hour_int", (col(time_col) / 100).cast("int"))

# Α) DataFrame API without UDF
print("\nParquet DataFrame API without UDF")
start_df = time.time()

df_parts = df_cleaned.withColumn(
    "day_part",
    when((col("hour_int") >= 5) & (col("hour_int") < 12), "MORNING")
    .when((col("hour_int") >= 12) & (col("hour_int") < 17), "AFTERNOON")
    .when((col("hour_int") >= 17) & (col("hour_int") < 21), "EVENING")
    .otherwise("NIGHT")
)

res_df = df_parts.groupBy("day_part").agg(
    count("*").alias("total_crimes"),
    count(when(col(premise_col) == "STREET", 1)).alias("street_crimes")
).withColumn("street_percentage", (col("street_crimes") / col("total_crimes")) * 100) \
 .orderBy(col("street_percentage").desc())

res_df.show()
time_df_val = time.time() - start_df
print(f"Time DataFrame without UDF (PARQUET): {time_df_val:.2f} seconds")


# Β) DataFrame API (with UDF)
print("\nParquet DataFrame API with UDF")
start_udf = time.time()

def get_day_part(hr):
    if hr is None: return "NIGHT"
    if 5 <= hr < 12: return "MORNING"
    elif 12 <= hr < 17: return "AFTERNOON"
    elif 17 <= hr < 21: return "EVENING"
    else: return "NIGHT"

day_part_udf = udf(get_day_part, StringType())

df_with_udf = df_cleaned.withColumn("day_part", day_part_udf(col("hour_int")))

res_udf = df_with_udf.groupBy("day_part").agg(
    count("*").alias("total_crimes"),
    count(when(col(premise_col) == "STREET", 1)).alias("street_crimes")
).withColumn("street_percentage", (col("street_crimes") / col("total_crimes")) * 100) \
 .orderBy(col("street_percentage").desc())

res_udf.show()
time_udf_val = time.time() - start_udf
print(f"Time DataFrame με UDF (PARQUET): {time_udf_val:.2f} seconds")


# C) RDD API
print("\nParquet RDD API")
start_rdd = time.time()

rdd = df_cleaned.rdd

def rdd_map(row):
    hour = row["hour_int"]
    premise = row[premise_col]
    
    if hour is None: dp = "NIGHT"
    elif 5 <= hour < 12: dp = "MORNING"
    elif 12 <= hour < 17: dp = "AFTERNOON"
    elif 17 <= hour < 21: dp = "EVENING"
    else: dp = "NIGHT"
    
    is_street = 1 if premise == "STREET" else 0
    return (dp, (1, is_street))

rdd_res = rdd.map(rdd_map) \
             .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
             .map(lambda x: (x[0], x[1][0], x[1][1], (x[1][1] / x[1][0]) * 100))

sorted_rdd = sorted(rdd_res.collect(), key=lambda x: x[3], reverse=True)


print("|day_part|total_crimes|street_crimes|street_percentage|")

for row in sorted_rdd:
    print(f"|{row[0]:>12}|{row[1]:>12}|{row[2]:>13}|{row[3]:>16.4f}%|")
print("+------------+------------+-------------+-----------------+")

time_rdd_val = time.time() - start_rdd
print(f"Time RDD API (PARQUET): {time_rdd_val:.2f} seconds")

spark.stop()