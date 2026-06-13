import time
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder \
    .appName("LA_Crime_Query3_Final") \
    .getOrCreate()
sc = spark.sparkContext

# Paths των αρχείων στο HDFS
geojson_path = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/data/LA_Census_Blocks_2020.geojson"
income_path = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/data/LA_income_2021.csv"


# ΥΛΟΠΟΙΗΣΗ 1: DATAFRAME API

print("\nRunning Query 3 with DataFrame API")
start_df_time = time.time()

df_geo = spark.read.option("multiline", "true").json(geojson_path).repartition(12)

df_population = df_geo.select(F.explode("features").alias("feature")) \
                      .select(
                          F.col("feature.properties.ZCTA20").alias("zip_code"),
                          F.col("feature.properties.POP20").cast("long").alias("population")
                      )

df_pop_grouped = df_population.groupBy("zip_code").agg(F.sum("population").alias("total_population"))

df_income = spark.read.option("delimiter", ";").option("header", "true").csv(income_path)

df_income_clean = df_income.select(
    F.col("Zip Code").alias("zip_code"),
    F.regexp_replace(F.col("Estimated Median Income"), r"[$, ]", "").cast("double").alias("median_income")
).filter(F.col("zip_code").isNotNull() & F.col("median_income").isNotNull())


# RUN 1: Default / BROADCAST 
df_joined = df_pop_grouped.join(df_income_clean, on="zip_code", how="inner")

# RUN 2: SORT MERGE JOIN
#df_joined = df_pop_grouped.join(df_income_clean.hint("MERGE"), on="zip_code", how="inner")

#  RUN 3: SHUFFLE HASH JOIN
#df_joined = df_pop_grouped.join(df_income_clean.hint("SHUFFLE_HASH"), on="zip_code", how="inner")

# RUN 4: SHUFFLE REPLICATE NL (NESTED LOOP) ---
#df_joined = df_pop_grouped.join(df_income_clean.hint("SHUFFLE_REPLICATE_NL"), on="zip_code", how="inner")



df_final = df_joined.withColumn(
    "income_per_capita", 
    F.col("median_income") / F.col("total_population")
).filter((F.col("total_population") > 0) & (F.col("income_per_capita").isNotNull())) \
 .select("zip_code", "total_population", "income_per_capita") \
 .orderBy(F.col("income_per_capita").desc())

print("[DataFrame]")
df_final.show(1000, False)

df_api_time = time.time() - start_df_time
print(f"DataFrame API Execution Time: {df_api_time:.2f} seconds")


# ΥΛΟΠΟΙΗΣΗ 2: RDD API 

print("\nRunning Query 3 with RDD API")
start_rdd_time = time.time()


rdd_pop = df_pop_grouped.rdd.map(lambda row: (str(row["zip_code"]).strip(), int(row["total_population"])))
rdd_income = df_income_clean.rdd.map(lambda row: (str(row["zip_code"]).strip(), float(row["median_income"])))

# Join των RDDs
rdd_joined = rdd_pop.join(rdd_income)

#  Υπολογισμός εισοδήματος
rdd_calculated = rdd_joined.map(lambda x: (
    x[0],     # zip_code
    x[1][0],  # total_population
    x[1][1] / x[1][0] if x[1][0] > 0 else 0  # median_income / total_population
)).filter(lambda x: x[1] > 0)

# collect
rdd_all = rdd_calculated.sortBy(lambda x: x[2], ascending=False).collect()

for row in rdd_all:
   print(f"|{row[0]:<8}|{row[1]:<18}|{row[2]:<17}|")
print(f"+--------+------------------+-----------------+")

rdd_api_time = time.time() - start_rdd_time
print(f"RDD API Execution Time: {rdd_api_time:.2f} seconds")

spark.stop()