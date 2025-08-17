# Databricks notebook source
# MAGIC %md
# MAGIC # Purpose: Create a RAG for financial data

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType
# from config import root_location, catalog_name, schema_name, table_name, endpoint_name, index_name

# COMMAND ----------

# MAGIC %md
# MAGIC # Initialize Vector Search on Databricks

# COMMAND ----------

# Sample financial documents.
financial_documents = [
        "The 2024 annual report for Company A shows a net profit of $50 million, a 15% increase from the previous year.",
        "Company B's stock price experienced a significant drop following the announcement of new government regulations on its industry.",
        "Economic forecasts predict a slight recession in the third quarter of 2025, which may affect consumer spending.",
        "The Q1 2024 earnings call for Company C highlighted a strategic shift towards renewable energy investments.",
        "A recent market analysis suggests that the technology sector will see moderate growth over the next five years."
    ]

# COMMAND ----------

# Define the schema for the DataFrame.
schema = StructType([
    StructField("id", StringType(), True),
    StructField("text", StringType(), True)
])

# Combine documents and IDs into a list of tuples for the DataFrame.
ids = [f"doc_{i}" for i in range(len(financial_documents))]

data = list(zip(ids, financial_documents))
spark_df = spark.createDataFrame(data, schema)

# COMMAND ----------

full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
spark.sql(
  f"CREATE CATALOG IF NOT EXISTS {catalog_name} MANAGED LOCATION '{root_location}'"
)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")

# COMMAND ----------

spark_df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(full_table_name)