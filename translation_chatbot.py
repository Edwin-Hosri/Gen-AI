# -*- coding: utf-8 -*-
"""translation Chatbot

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NjQkJVF8EvhMfvN5iVDAyiYy37DAjzLW
"""

!pip install langchain-huggingface

"""Installing Dependencies"""

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import MarianMTModel, MarianTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import os

"""HuggingFace login setup"""

from huggingface_hub import login
login(token="YOUR_HUGGING_FACE_TOKEN")

models={
    "1": "Helsinki-NLP/opus-mt-en-es", #spanish
    "2": "Helsinki-NLP/opus-mt-en-fr", #french
    "3": "Helsinki-NLP/opus-mt-en-zh", #chinese
    "4": "Helsinki-NLP/opus-mt-en-ru", #russian
    "5": "Helsinki-NLP/opus-mt-en-mr", #marathi
    "6": "Helsinki-NLP/opus-mt-en-ar", #arabic
    "7": "Helsinki-NLP/opus-mt-en-de", #german
}

languages=['Spanish','French','Chinese','Russian','Marathi','Arabic','German']


def get_translation(keyNumber,text):

    #model loading and setup
    model_name = models[keyNumber]
    tokenizer= MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    #pipeline setup
    device = 0 if torch.cuda.is_available() else -1
    transformer_pipeline = pipeline('translation',
                     model=model,
                     tokenizer=tokenizer,
                     max_length=200,
                     truncation=True,
                     temperature= 0.6,
                     device=device,
                    )

    #pipeline using HuggingFace from langchain
    hf_pipeline = HuggingFacePipeline(pipeline=transformer_pipeline)

    #writting prompt for LLM
    template = f" You should be the translator and translate the following text to {languages[int(keyNumber)-1]}: '{text}' "

    prompt = PromptTemplate(input_variables=['text'],
                        template=template
                        )

    llm_chain = LLMChain(
                         prompt=prompt,
                         llm=hf_pipeline
                        )

    llm_chain.run({'text': input_text}) #the text is processed and translated
    translated_text = transformer_pipeline(text)[0]['translation_text'] #transformer_pipeline(text) returns a list of dictionaries.
    return translated_text

if __name__ == "__main__":
  while True:
      translated_language_num = input("""
      1- Spanish\n
      2- French\n
      3- Chinese\n
      4- Russian\n
      5- Marathi\n
      6- Arabic\n
      7- German\n
      <Write any other key to exit>
      Enter the number of the language you want to translate to:
      """)
      if translated_language_num not in models:
          print('Terminated')
          break

      input_text = input(' Enter the text to translate: ')

      print(get_translation(translated_language_num,input_text))
