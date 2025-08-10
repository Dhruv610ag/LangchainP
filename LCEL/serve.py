from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check for API key
groq_api_key = os.getenv("GROQ_API_KEY")
assert groq_api_key, "Missing GROQ_API_KEY in .env file"#raise an error if the api key is not found 

# Initialize model (corrected model name)
model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key) #type:ignore # take the model from the docs from the original side of what you are using 

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}:"),#tells the system what to do 
    ("user", "{text}")# take the input from the user that the system will take the input from the users
])

# Output parser
parser = StrOutputParser()

# Create chain
chain = prompt_template | model | parser 

## App definition
app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces")

## Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

#local host system 
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=4000)

