from dotenv import load_dotenv
from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
from langchain_huggingface import HuggingFacePipeline
import os
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Generation config
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.6,
    "top_k": 40,
    "top_p": 0.9,
}

login(HF_TOKEN)

model_name = "google/pegasus-xsum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

"""
pipe = pipeline("summarization",
                model=loaded_model, 
                tokenizer=tokenizer,
                device = 0 if cuda.is_available() else -1,
                temperature = 0.6,
                top_k = 40,
                top_p = 0.9,
                )
"""

#more efficient way to do the same thing when it comes to configurations


pipe = pipeline(
    "summarization",
    model=loaded_model,
    tokenizer=tokenizer,
    device= 0 if cuda.is_available() else -1,
    **GENERATION_CONFIG
)
#The ** syntax in Python is called dictionary unpacking (or sometimes "keyword argument unpacking").It allows you to pass all key-value pairs in a dictionary as named arguments to a function. (exactly similar to the first commented out pipeline initialization)

pipeline = HuggingFacePipeline(
    pipeline = pipe,
)

prompt = PromptTemplate(
    input_variables = ["input"],
    template = """
         Summarize the following text:
         {input}
    """
)

chain = prompt | pipeline 
#this is the same as LLMChain(llm = pipeline, prompt = prompt) ORDER MATTERS, we pass prompt first and then pipeline because we want to pass the input to the prompt first and then the prompt to the pipeline


if __name__ == "__main__":
    print("Welcome to the summarization chatbot!")
    while True:
        input_text = input("Enter the text to summarize: ")
        if not input_text:
            print("No input text provided. Exiting...")
        else:
            result = chain.invoke({"input": input_text})
            print("Summary:", result['text'] if isinstance(result, dict) else result)
         
        retry = input("Do you want to continue? (y/n): ")
        if retry == "n":
            break
