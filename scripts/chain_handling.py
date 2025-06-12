#%% packages
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#%% model instance
MODEL_NAME = "llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME, 
                 temperature=1)
# %% prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a writer of jokes and provide the funniest ones. You answer in {output_language}."),
    ("user", "Tell me a joke about <<{topic}>>. The scene is described as <<{scene_description}>>")
])

# %% chain 
chain = prompt_template | model | StrOutputParser()

# %% invoke chain
from pprint import pprint
pprint(chain.invoke({'topic': 'Germans',
                     'output_language': 'German',
                     'scene_description': 'ordering a beer'}),  width=50)

# additional variables: output_language, scene_description
# %%
