# Databricks notebook source
# MAGIC %md
# MAGIC # Purpose: Ingest and Store data into Bronze Table

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit

# COMMAND ----------

def set_prompt_in_bronze_table(user_prompt: str) -> None:

    # Create a DataFrame from the user prompt and a timestamp
    prompt_df = spark.createDataFrame(
        [
            (user_prompt,)
        ],
        ["prompt_text"]
    ).withColumn("ingestion_timestamp", current_timestamp())

    # Write the DataFrame to the Bronze Delta table (raw)
    prompt_df.write \
        .format("delta") \
        .mode("append") \
        .save(bronze_table_path)
    