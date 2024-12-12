# Generative AI Projects

## Q&A Chatbot

### Introduction
A chatbot that answers user queries based on predefined knowledge.

### Environment Variable
1. Register/Login to HuggingFace.
2. Generate HuggingFace Token.
3. Apply permission to access Llama models (Meta).
4. Open Google Colab.
5. Create a new secret key:
   - Give the key a name and paste the generated token string into the value's part.
6. Import the key from Google Colab:
   ```python
   from google.colab import userdata
   userdata.get('Name_of_ur_key')


## Translating Chatbot

### Introduction
A chatbot that translates text from English to multiple languages in real-time.

### Environment Variable
1. Register/Login to HuggingFace.
2. Generate HuggingFace Token.
3. Open Google Colab.
4. Create a new secret key:
   - Give the key a name and paste the generated token string into the value's part.
5. Import the key from Google Colab:
   ```python
   from google.colab import userdata
   userdata.get('Name_of_ur_key')


## IMPORTANT: These Programs should run on Google colab. 
