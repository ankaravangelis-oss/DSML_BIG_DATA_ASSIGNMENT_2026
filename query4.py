import time
from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("LA_Crime_Query4_SQL_Steps") \
    .getOrCreate()

# Διαδρομές HDFS
crimes_path_2010_2019 = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/user/dsml00292/LA_Crime_Data_Parquet/LA_Crime_Data_2010_2019.parquet"
crimes_path_2020_2025 = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/user/dsml00292/LA_Crime_Data_Parquet/LA_Crime_Data_2020_2025.parquet"
stations_path = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/data/LA_Police_Stations.csv"

print("Ανάγνωση δεδομένων από το HDFS")
# ΑΝΑΓΝΩΣΗ PARQUET 
df_crimes_1 = spark.read.parquet(crimes_path_2010_2019)
df_crimes_2 = spark.read.parquet(crimes_path_2020_2025)
df_crimes = df_crimes_1.union(df_crimes_2)

df_PD_stations = spark.read.csv(stations_path, header=True, inferSchema=False)

#Temp Views
df_crimes.createOrReplaceTempView("raw_crimes")
df_PD_stations.createOrReplaceTempView("raw_stations")


start_sql_time = time.time()
#/*+ BROADCAST(s) */
#/*+ MERGE(c, s) */
#/*+ SHUFFLE_HASH(c, s) */
#/*+ SHUFFLE_REPLICATE_NL(c, s) */

sql_step1 = """
    SELECT /*+ BROADCAST(s) */
        c.`DR_NO` as crime_id,
        s.DIVISION as station_division,
        6371.0 * 2.0 * ASIN(SQRT(
            POWER(SIN(RADIANS(CAST(s.Y AS FLOAT) - CAST(c.LAT AS FLOAT)) / 2.0), 2) +
            COS(RADIANS(CAST(c.LAT AS FLOAT))) * COS(RADIANS(CAST(s.Y AS FLOAT))) *
            POWER(SIN(RADIANS(CAST(s.X AS FLOAT) - CAST(c.LON AS FLOAT)) / 2.0), 2)
        )) as distance_km
    FROM raw_crimes c
    CROSS JOIN raw_stations s
    WHERE c.LAT IS NOT NULL AND c.LON IS NOT NULL 
      AND CAST(c.LAT AS FLOAT) != 0.0 AND CAST(c.LON AS FLOAT) != 0.0
      AND s.X IS NOT NULL AND s.Y IS NOT NULL
      AND CAST(s.X AS FLOAT) != 0.0 AND CAST(s.Y AS FLOAT) != 0.0
"""

distance_df = spark.sql(sql_step1)
distance_df.createOrReplaceTempView("distance_table")

sql_step2 = """
    SELECT 
        crime_id,
        station_division,
        distance_km,
        ROW_NUMBER() OVER (PARTITION BY crime_id ORDER BY distance_km ASC) as rnk
    FROM distance_table
"""

ranked_df = spark.sql(sql_step2)
ranked_df.createOrReplaceTempView("ranked_table")

sql_step3 = """
    SELECT 
        station_division as division,
        ROUND(AVG(distance_km), 3) as average_distance,
        COUNT(*) as `#`
    FROM ranked_table
    WHERE rnk = 1
    GROUP BY station_division
    ORDER BY `#` DESC
"""

sql_result = spark.sql(sql_step3)
sql_result.show(30, truncate=False)

sql_time = time.time() - start_sql_time
print(f"Query 4 SQL Execution Time: {sql_time:.2f} seconds")

# Optimizer 
print("Spark Optimizer SQL Execution Plan")
sql_result.explain()

spark.stop()