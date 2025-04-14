from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import YouTubeSearchTool
from langchain_huggingface import HuggingFacePipeline
from langchain_google_community import GoogleSearchResults
import torch
from huggingface_hub import login
from transformers import pipeline
from dotenv import load_dotenv
import os


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "gpt2"

# Getting runtime for GPU/CPU selection
runtime = 0 if torch.cuda.is_available() else -1

# Login using HuggingFace token
login(token=HF_TOKEN)

# Model selection
llm_pipeline = pipeline(
    'text-generation',
    model=MODEL,
    device=runtime,
    temperature=0.7,
    top_k=1,
    top_p=0.85,
    max_new_tokens=100,
    pad_token_id=50256,
)

# Wrap the pipeline in HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize agent tools
tools = [
    Tool(
        name="LLM Generation",
        func= llm.invoke,
        description="Generates text using LLM",
    ),
    YouTubeSearchTool(),
    GoogleSearchResults(),
]

# Initialize memory for agent
memory = ConversationBufferWindowMemory(window_size=4)

template = """
    You are a agent chatbot and you need to answer client questions in clear and concise manner.
    You are free to use the tools provided to your advantage.
    Here is some context that could be relevant to your answer:\n\n {history} \n\n
    Client Question: {input}
    agent chatbot: 
"""

prompt = PromptTemplate(template=template, input_variables=["history", "input"])

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True,
    handle_parsing_errors=True,
)

while True:
    user_input = input("Client Question: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Terminated")
        break
    try:
        response = agent.invoke(input=user_input)
        print(f"agent chatbot: {response}")
    except Exception as e:
        print(f"Error: {e}")
        
    torch.cuda.empty_cache()
    
