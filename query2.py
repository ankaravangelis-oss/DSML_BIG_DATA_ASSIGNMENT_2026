import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


spark = SparkSession.builder \
    .appName("LA_Crime_Query2") \
    .getOrCreate()

path_1 = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv"
path_2 = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"

df_1 = spark.read.csv(path_1, header=True, inferSchema=False)
df_2 = spark.read.csv(path_2, header=True, inferSchema=False)
df_all = df_1.union(df_2)


parsed_date = F.coalesce(
    F.to_date(F.col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a"),
    F.to_date(F.col("DATE OCC"), "yyyy MMM dd hh:mm:ss a")
)

df_with_time = df_all.withColumn("year", F.year(parsed_date)) \
                     .withColumn("month", F.month(parsed_date))

# 
df_filtered = df_with_time.filter(F.col("year").isNotNull() & F.col("month").isNotNull())


# DATAFRAME API

print("\n Running Query 2 with DataFrame API")
start_df_time = time.time()


df_grouped = df_filtered.groupBy("year", "month").agg(F.count("*").alias("crime_total"))


window_spec = Window.partitionBy("year").orderBy(F.col("crime_total").desc())

df_ranked = df_grouped.withColumn("ranking", F.dense_rank().over(window_spec)) \
                        .filter(F.col("ranking") <= 3)

df_final_result = df_ranked.orderBy(F.col("year").asc(), F.col("crime_total").desc(), F.col("ranking").asc())

df_final_result.show(48)

df_time = time.time() - start_df_time
print(f"DataFrame API Execution Time: {df_time:.2f} seconds")


#  SQL API 
print("\n Running Query 2 with SQL")

start_sql_time = time.time()

df_filtered.createOrReplaceTempView("crimes")

sql_step1 = """
    SELECT 
        year,    month,    COUNT(*) as total_crimes,   DENSE_RANK() OVER (PARTITION BY year ORDER BY COUNT(*) DESC)   as ranking
    FROM crimes
    GROUP BY year, month
"""
new_df = spark.sql(sql_step1)
new_df.createOrReplaceTempView("new_table")


sql_step2 = """
    SELECT year, month, total_crimes, ranking
    FROM new_table
    WHERE ranking <= 3
    ORDER BY year ASC, total_crimes DESC
"""

sql_result = spark.sql(sql_step2)
sql_result.show(48)

sql_time = time.time() - start_sql_time
print(f"SQL API Time: {sql_time:.2f} seconds")

spark.stop()


