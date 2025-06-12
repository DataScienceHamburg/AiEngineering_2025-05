#%% packages
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama 
# %% model instance
# MODEL_NAME="gemma2-9b-it"
# model = ChatGroq(model=MODEL_NAME)
MODEL_NAME="qwen3:0.6b"
model = ChatOllama(model=MODEL_NAME, 
                   temperature=0.2)


# %% model inference
user_query = "What do know about Hamburg?"
res = model.invoke(user_query)
res

#%%
pprint(res.content)
# %%
res.model_dump()

# %%
