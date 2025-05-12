#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %% prompt template
messages = [
    ("system", "You are an AI assistant that translates English into another language."),
    ("user", "Translate the sentence <<{input}>> into <<{target_language}>>.")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt_template

#%% model via Groq
MODEL_NAME = "llama3-8b-8192"
model = ChatGroq(model=MODEL_NAME)

#%% Chain definition
chain = prompt_template | model

#%% inference
user_input = {"input": "Hallo Welt!", "target_language": "French"}
res = chain.invoke(user_input)
#%%
res.content