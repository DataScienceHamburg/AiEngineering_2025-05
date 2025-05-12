#%% packages
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %%
PROMPT_NAME = "hardkothari/prompt-maker"
prompt_template = hub.pull(PROMPT_NAME)

#%%
MODEL_NAME = "llama-3.2-1b-preview"
model = ChatGroq(model=MODEL_NAME)

# %% chain setup
chain = prompt_template | model | StrOutputParser()

# %% chain invocation
lazy_prompt = "a human in the center, and a neural network interact; futuristic design"
task = "a prompt for text-to-image algorithm to create an image"
res = chain.invoke({"lazy_prompt": lazy_prompt, "task": task})

#%%
from pprint import pprint
from pyperclip import copy
copy(res)
pprint(res)
# %%
