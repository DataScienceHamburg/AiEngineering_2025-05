#%%
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
# %%
load_dotenv(find_dotenv(usecwd=True))
#%% show API key
os.getenv('GROQ_API_KEY')

# %%
MODEL_NAME = "deepseek-r1-distill-qwen-32b"
model = ChatGroq(model=MODEL_NAME,
                 api_key=os.getenv('GROQ_API_KEY'),
                 temperature=0.2
                 )
# %% model inference
user_query = "Was ist Bildungsurlaub?"
res = model.invoke(user_query)

#%% model output
from pprint import pprint
pprint(res.content)

#%%
res.model_dump()

#%% cost estimate
input_token_cost = 0.15
output_token_cost = 0.60
9 / 1E6 * input_token_cost + 456 /1E6 * output_token_cost