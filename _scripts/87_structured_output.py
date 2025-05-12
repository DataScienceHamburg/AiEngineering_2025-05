#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pydantic import BaseModel
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser

#%% Idea: 

# output
# { "recipe_name": "...", "ingredients": [{"name": "...", "quantity": ...}], "steps": []}


#%% define output format
class RecipeOutput(BaseModel):
    recipe_name: str
    ingredients: list[dict[str, str]]  # Change float to str for quantity
    steps: list[str]

    
# %% prompt template
messages = [
    ("system", "You are a recipe expert and deliver key information on specific recipes."),
    ("user", "Please provide key information on the recipe <<{recipe}>>. First translate everything into {language}. Return the result as JSON with the keys recipe_name, ingredients as list of dicts with keys name and quantity, steps as list of strings.")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt_template

#%% model via Groq
MODEL_NAME = "llama3-8b-8192"
model = ChatGroq(model=MODEL_NAME)

#%% Chain definition
chain = prompt_template | model | StrOutputParser()| SimpleJsonOutputParser(pydantic_object=RecipeOutput)

#%% inference
input_text = """
Her er en opskrift på Spaghetti Carbonara:
Du skal bruge 200 g spaghetti, 100 g bacon, 2 æg, 50 g parmesan og lidt peber.
Kog spaghetti, steg bacon, bland æg og ost, og bland det hele med de varme nudler.
"""
user_input = {"recipe": input_text, "language": "English"}
res = chain.invoke(user_input)
#%%
res
# %%
