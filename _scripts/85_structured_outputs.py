#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pydantic import BaseModel
from langchain_core.output_parsers import SimpleJsonOutputParser


#%% Idea: 
# model as book expert
# output: {"title": "...", "author": "...", "summary": "..."}

#%% define output format
class BookOutput(BaseModel):
    title: str
    author: str
    summary: str


# %% prompt template
messages = [
    ("system", "You are a book expert and deliver key information on specific books."),
    ("user", "Please provide key information on the book <<{book}>>. Return the result as JSON with the keys title, author, summary.")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt_template

#%% model via Groq
MODEL_NAME = "llama3-8b-8192"
model = ChatGroq(model=MODEL_NAME)

#%% Chain definition
chain = prompt_template | model | SimpleJsonOutputParser(pydantic_object=BookOutput)

#%% inference
user_input = {"book": "1984"}
res = chain.invoke(user_input)
#%%
res
# %%
