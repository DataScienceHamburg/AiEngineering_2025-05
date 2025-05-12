#%% package
from langchain_ollama import ChatOllama

#%%
MODEL_NAME = "gemma3:1b"
model = ChatOllama(model=MODEL_NAME)
# %% model inference

user_query = "What is Hamburg?"
res = model.invoke(user_query)

# %%
res.content
# %%
from pprint import pprint
pprint(res.content)
# %%
