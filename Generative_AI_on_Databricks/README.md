# Generative AI RAG on Databricks
This project implements a scalable Retrieval-Augmented Generation (RAG) pipeline on the Databricks platform. It is designed to ingest a user's prompt, retrieve relevant financial documents, and use a large language model (LLM) to generate a well-informed response.

The pipeline follows the recommended Medallion Architecture, separating data into three distinct layers: Bronze (raw data), Silver (enriched data), and Gold (final, aggregated data).

## Project Architecture
- Bronze Layer (bronze.py): This is the raw data ingestion layer. It captures and stores every user prompt as-is in a Delta table. This ensures a complete, untransformed history of all inputs.
- Silver Layer (silver.py): The transformation and enrichment layer. It reads the raw prompts from the Bronze layer and generates vector embeddings. These embeddings are stored in a new Delta table, which serves as the source for the Databricks Vector Search.
- Gold Layer (gold.py): The consumption layer. This script uses a RAG agent to retrieve relevant context from the Silver layer and generates a final, coherent response from the LLM. The entire interaction, including the original prompt, retrieved context, and the final response, is logged in a Delta table for auditing and analysis.

## Setup Instructions
### Step 1: Create a GPU Cluster on Databricks
To run this project, you need a cluster with GPU capabilities to handle the LLM inference and embedding generation efficiently.

1. Navigate to the **Compute** section in the Databricks sidebar.
2. Click on **Create Cluster**.
3. Choose a **Node Type** with GPU support (e.g., a "Machine Learning" node type with an attached GPU instance).
4. If you encounter a QuotaExceeded error, you will need to request a quota increase for your Azure subscription in the relevant region.

### Step 2: Install Libraries
The project relies on several key Python libraries. You should install them by attaching the `requirements.txt` file to your cluster.

1. Navigate to the **Libraries** tab on your cluster configuration page.
2. Click **Install New**.
3. Select **Workspace**.
4. Upload the `requirements.txt` file from the project repository.
5. Click **Install**.

### Step 3: Create a Vector Search Index
After your data is prepared in the Silver layer's Delta table, you need to create a Vector Search Index to enable efficient similarity search.

1. **Navigate to the Catalog**: In the Databricks sidebar, click on the **Catalog** icon.
2. **Find Your Delta Table**: In the Catalog Explorer, locate the Delta table you created from the RAG data preparation script (e.g., finance_rag_data).
3. **Create the Index**: Click the **Create** button in the top right and select Vector Search Index.
4. **Configure the Index**: Fill in the form with the following details:
- **Index Name**: Provide a unique name for your index, such as `rag.finance.finance_rag`.
- **Endpoint**: Select the endpoint you created in the Compute section. If you haven't created one, you can do it from this page.
- **Primary Key**: Choose the column you designated as the primary key (`id`).
- **Embedding Source**: Select **Compute embeddings**.
- **Text Column**: Choose the column containing your text content, which is typically `text`.
- **Embedding Model**: Select a model that matches the one used in your `silver.py script`, which is `sentence-transformers/bert-large-nli-max-tokens`.
5. Click **Create**. Databricks will now handle the process of creating the index and populating it with embeddings.

### Step 4: Secure Your API Tokens
**Never hard-code your tokens in your code**. This project uses Databricks Secrets to securely manage API credentials for Hugging Face.

1. **Create a Secret Scope**: Open your local terminal and use the Databricks CLI to create a secret scope.
```
# Create the scope
databricks secrets create-scope --scope gen_ai_scope
```
2. **Store Your Hugging Face Token**: Use the `databricks secrets put` command. This will open a text editor in your terminal where you can paste the token. After pasting, type `:wq` and press `Enter` to save and exit.
```
# Put your token into the scope. The CLI will prompt you for the value.
databricks secrets put --scope gen_ai_scope --key huggingface_token
```
3. **Store Your Databricks Token**: Use the same command for your Databricks token, which will be used for Vector Search. This will also open the text editor where you should paste the token and then save and exit with `:wq` and `Enter`.
```
databricks secrets put --scope rag_scope --key databricks_token
```

### Step 5: Run the Pipeline
With the cluster and secrets configured, you can now run the pipeline from the main `run.py` notebook.

1. **Open the `run.py` notebook** in your Databricks workspace.
2. The notebook creates a widget for you to enter your prompt.
3. Run the notebook. It will execute the `bronze`, `silver`, and `gold` notebooks in sequence, handling the entire ETL process and generating a final response.

### Recommended Optimizations
For production use cases with millions of inputs, consider these optimizations:
- **Z-Ordering on Silver Layer**: After generating embeddings, run `OPTIMIZE` on the Silver table with `ZORDER BY (text_embedding)`. This groups similar vectors together, dramatically speeding up vector search queries.
- **Databricks Vector Search**: Instead of a manual similarity search in the Silver layer, use the native Databricks Vector Search API for a fully managed, scalable, and highly performant solution.