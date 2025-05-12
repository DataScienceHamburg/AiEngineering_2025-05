#%% Idea
# Model as book expert
# Output: {"title": "...", "author": "...", "summary": "..."}
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from langchain_core.output_parsers import SimpleJsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, validator
 
sales_info ="""
Hier sind die Verkaufszahlen für diese Woche: Montag: 150, Dienstag: 200, Mittwoch: 180, Donnerstag: 220, Freitag: 50. Wir arbeiten von Mo-Fr.
"""
 
#%% Define output format
class SalesOutput(BaseModel):
    D01: str
    D02: str
    D03: str
    D04: str
    D05: str

    

   
 
#%% Load API Key
load_dotenv(find_dotenv(usecwd=True))
#%% Show API Key
import os
x=os.getenv("GROQ_API_KEY")
print(x)
from langchain_groq import ChatGroq
 
MODEL_NAME = "deepseek-r1-distill-qwen-32b"
model = ChatGroq(model=MODEL_NAME)
 
 
#%% Prompt Template
messages = [
    ("system", "You are a ales accountant and deliver a weekly summary"),
    ("user", "Please provide summary: <<{sales}>>. Return the result as JSON with the keys for the weekdays: 'D01', 'D02', 'D03', 'D04', 'D05'" )  
]
 
prompt_template=ChatPromptTemplate.from_messages(messages=messages)
 
chain = prompt_template | model | SimpleJsonOutputParser(pydantic_object=SalesOutput)
 
user_input= {"sales": sales_info}
res = chain.invoke(user_input)
print(res)
#%%
res.content
 