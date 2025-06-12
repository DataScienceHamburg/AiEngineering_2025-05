#%% packages
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel


#%% model instance
MODEL_NAME = "llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME, 
                 temperature=1)

#%% expected output:
# {'title': '...' ,'author': '...', 'summary': '...'}
class BookOutput(BaseModel):
    title: str
    author: str
    summary: str

# %% prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "you are a book expert and deliver key info on specific books. return the result as json with the keys title, summary, and author. only return one specific book, never multiple. select the most famouse one."),
    ("user", "please provide key information on the book: {book_description}")
])

# %% chain 
chain = prompt_template | model | JsonOutputParser()

# %% invoke chain
from pprint import pprint
pprint(chain.invoke({'book_description': 'a superintelligent ai commits murder'}),  width=50)

# additional variables: output_language, scene_description
# %%
