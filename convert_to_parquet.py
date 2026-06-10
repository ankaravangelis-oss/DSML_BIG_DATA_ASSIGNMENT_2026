from pyspark.sql import SparkSession

# Αρχικοποίηση SparkSession
spark = SparkSession.builder \
    .appName("LA_Crime_Convert_CSV_to_Parquet") \
    .getOrCreate()

# HDFS Paths
path_1_csv = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv"
path_2_csv = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"

# Parquet Paths
path_1_pq = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/user/dsml00292/LA_Crime_Data_Parquet/LA_Crime_Data_2010_2019.parquet"
path_2_pq = "hdfs://hdfs-namenode.default.svc.cluster.local:9000/user/dsml00292/LA_Crime_Data_Parquet/LA_Crime_Data_2020_2025.parquet"

print("\nCSV 2010-2019 and writing as Parquet")
df_1 = spark.read.csv(path_1_csv, header=True, inferSchema=True)
df_1.write.mode("overwrite").parquet(path_1_pq)

print("\nCSV 2020-2025 and writing as Parquet")
df_2 = spark.read.csv(path_2_csv, header=True, inferSchema=True)
df_2.write.mode("overwrite").parquet(path_2_pq)

spark.stop()