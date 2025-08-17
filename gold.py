# Databricks notebook source
# MAGIC %md
# MAGIC # Purpose: Store all reponse data and query into Gold Table

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import torch
import transformers
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool

# COMMAND ----------

class LLMAnalyzer:
    def __init__(self, HF_API: str, model_id = llm_model) -> None:
        self.api = HF_API
        self.model_id = llm_model

        self.model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            token=self.api
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            config=self.model_config,
            device_map='auto',
            token=self.api,
            low_cpu_mem_usage=True, # if true, create multiple processes or threads to handle loading and offloading -> slow
            weights_only=True
        )

        # Set device
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            add_eos_token=True,  
            trust_remote_code=True,
            token=self.api,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.5,
        )

        self.llm = HuggingFacePipeline(pipeline=self.generator)

# COMMAND ----------

def __ai_agent(llm_agent, user_prompt: str, rag_context: str):
    def search_web_news(query: str) -> str:
        """Performs a DuckDuckGo news search for the given query, daily results, max 2."""
        wrapper = DuckDuckGoSearchAPIWrapper(time="d", max_results=2)
        engine = DuckDuckGoSearchResults(api_wrapper=wrapper, backend="news")
        return engine.invoke(f"{query}")
    
    tool_search_news = Tool(
        name="news_search",
        func=search_web_news,
        description="Tool to perform a DuckDuckGo news search. "
                    "Useful for current events or recent information. "
                    "Input should be a search query string. Returns up to 2 news results."
    )
       
    tools = [tool_search_news]

    prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a highly knowledgeable finance chatbot. Your purpose is to provide accurate, insightful,
            and actionable financial advice to users, tailored to their specific needs and contexts.

            Available tools: {tools}
            Available tool names: {tool_names}

            Additional context for your answer: {rag_context}
            
            Responses should always follow this format:
            Question: The question the user wants to answer
            Thought: Think about what to do to answer the question
            Action: The tool to use (must be one of the available tools)
            Action Input: Input to the tool
            Observation: Result of the tool
        
            ...(Thought/Action/Action Input/Observation can be repeated as many times as necessary to answer the question)
            Thought: Determine that it's time to provide the final answer to the user
            Final Answer: The final answer to the user
            """),

            ("user", "Analyze this company's finance status: {user_prompt}\n{agent_scratchpad}")
        ])

    agent = create_react_agent(llm_agent.llm, tools, prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Keep verbose=True to ensure intermediate steps are printed to stdout
        handle_parsing_errors=True
    )

    print("\n--- Agent's Reasoning Process ---")

    return agent_executor.invoke({"user_prompt": user_prompt, "rag_context": rag_context})

# COMMAND ----------

def set_gold_layer_with_ai_agent(llm_agent, user_prompt: str, rag_context: str) -> None:
    try:
        result = __ai_agent(llm_agent, user_prompt, rag_context)
        print(result)
        llm_response = result['output']
    except Exception as e:
        print(f"Error in set_gold_layer_with_ai_agent: {e}")
    
    # llm_response = result['output']
    final_prompt = result['intermediate_steps'][-1]

    print("\n--- Final Answer ---")
    print(result['output'])
    print("\n" + "=" * 80 + "\n")

    gold_df = spark.createDataFrame(
        [(user_prompt, rag_context, final_prompt, llm_response, current_timestamp())],
        ["user_query", "retrieved_context", "final_llm_prompt", "llm_response", "timestamp"]
    )
    
    gold_df.write \
        .format("delta") \
        .mode("append") \
        .save(gold_table_path)