# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %run ./bronze

# COMMAND ----------

# MAGIC %run ./silver

# COMMAND ----------

# MAGIC %run ./gold

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FinanceGenAI").getOrCreate()
llm_agent = LLMAnalyzer(get_huggingface_token())

# COMMAND ----------

dbutils.widgets.text("prompt", "", "Enter your prompt here")
user_prompt = dbutils.widgets.get("prompt")
if user_prompt:
    set_prompt_in_bronze_table(user_prompt)
    response = set_silver_table_with_embeddings(user_prompt, 2, 0.00001)
    set_gold_layer_with_ai_agent(llm_agent, user_prompt, response)
