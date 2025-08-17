# Databricks notebook source
# Define the paths for your data layers
bronze_table_path = 'dbfs:/mnt/gen_ai/bronze_table'
silver_table_path = 'dbfs:/mnt/gen_ai/silver_embeddings'
gold_table_path = 'dbfs:/mnt/gen_ai/golde_table'

# COMMAND ----------

# Define the index and endpoint name for RAG
root_location = "abfss://YOUR_PATH.dfs.core.windows.net/YOUR_NUM"
catalog_name = "rag"
schema_name = "finance"
index_name = "rag.finance.finance_rag"
endpoint_name = "finance_rag"

# COMMAND ----------

# Define the model for RAG to vectorize 
rag_model = "sentence-transformers/bert-large-nli-max-tokens"

# COMMAND ----------

# Define LLM model 
llm_model = "tarun7r/Finance-Llama-8B"

# COMMAND ----------

DATABRICKS_HOST = "https://Databricks_Home_.99999.azuredatabricks.net/"

def get_databricks_toke():
    # Use Databricks Secrets to securely retrieve the Databricks API token.
    # Step 1: Create a token on Databricks.
    # Step 2: Copy the token.
    # Step 3: paste the token like following lines.
    # Run: if the version is v0.264.2, pip install --upgrade databricks-cli and restart terminal on databricks
    # Example CLI command: `databricks secrets create-scope --scope gen_ai_scope`
    # Example CLI command: `databricks secrets put --scope gen_ai_scope --key databricks_token
    # Paste PASTED_TOKEN, then :wq + Enter to save the file
    # The `dbutils.secrets.get()
    try:
        return dbutils.secrets.get(scope='rag_scope', key='databricks_token')
    except NameError:
        print("dbutils is not defined. This code can only be run inside a Databricks notebook.")
        return None

# COMMAND ----------

def get_huggingface_token(): 
    # Use Databricks Secrets to securely retrieve the Hugging Face API key.
    # Step 1: Create a token on HuggingFace.
    # Step 2: Copy the token.
    # Step 3: paste the token like following lines.
    # Run: if the version is v0.264.2, pip install --upgrade databricks-cli and restart terminal on databricks
    # Example CLI command: `databricks secrets create-scope --scope gen_ai_scope
    # Example CLI command: `databricks secrets put --scope gen_ai_scope --key huggingface_token
    # Paste PASTED_TOKEN, then :wq + Enter to save the file
    # The `dbutils.secrets.get()` function retrieves the key from the specified scope.
    try:
        return dbutils.secrets.get(scope="gen_ai_scope", key="huggingface_token")
    except NameError:
        print("dbutils is not defined. This code can only be run inside a Databricks notebook.")
        return None