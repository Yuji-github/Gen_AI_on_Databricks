# Databricks notebook source
# MAGIC %md
# MAGIC # Purpose: Embedding Query and Store in Vector Database

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer
from pyspark.sql.functions import current_timestamp, lit, col
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG

# COMMAND ----------

# the correct 1024-dimension to match the index
model = SentenceTransformer(rag_model)

vsc = VectorSearchClient(
    workspace_url=DATABRICKS_HOST,
    personal_access_token=get_databricks_toke()
)

def __retrieve_and_augment(query_text, num_results:int, similarity_threshold: float) -> str:
    """
    Takes a user query, embeds it, and searches for similar documents
    in the Databricks Vector Search index.
    """
    try:
        # Generate 1024 dims embedding for the user's query.
        query_embedding = model.encode([query_text]).tolist()[0]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return ""
    
    # Search the Databricks Vector Search index.
    # The 'columns' parameter specifies which fields to return from the documents.
    # The 'num_results' parameter is similar to Chroma's 'n_results'.
    try:
        results = vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        ).similarity_search(
            query_vector=query_embedding,
            columns=["text"],
            num_results=num_results,
        )
    except Exception as e:
        print(f"Error performing search: {e}")
        return ""
    
    # The API response now returns a list of lists in 'data_array',
    # where the first element is the text and the second is the score.
    retrieved_documents = [
        d[0] for d in results.get('result', {}).get('data_array', [])
        if d[1] >= similarity_threshold
    ]

    # Format the retrieved documents as a single string to be used as context
    # for a large language model.
    context = "\n\n".join(retrieved_documents)
    return context

# COMMAND ----------

# to store vectorized query data for the future improvement
def __generate_embeddings(text: str) -> list:
    """
    Generates sentence embeddings for a given text using the Sentence-Transformer model.
    
    Args:
        text (str): The input text to embed.

    Returns:
        list: A list of floats representing the sentence embedding.
    """
    if text is None:
        return None
    return model.encode(text, normalize_embeddings=True).tolist()

# COMMAND ----------

def set_silver_table_with_embeddings(user_prompt: str, num_results:int = 2, similarity_threshold: float = 0.5) -> str:
    bronze_df = spark.read.format("delta").load(bronze_table_path)
    generate_embeddings_udf = udf(__generate_embeddings, ArrayType(FloatType()))

    # Apply the UDF to create a new `text_embedding` column in the Silver DataFrame.
    silver_df = bronze_df.withColumn("text_embedding", generate_embeddings_udf(col("prompt_text")))
    silver_df.write.format("delta").mode("overwrite").save(silver_table_path)

    # Z-Ordering to speed up vector search
    spark.sql(f"OPTIMIZE delta.`{silver_table_path}` ZORDER BY (text_embedding)")

    return __retrieve_and_augment(user_prompt, num_results, similarity_threshold)
