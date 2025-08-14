import streamlit as st
# import openai
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to all the user queries and Give Answer in the elaborated and well defined Manner"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm,temperature,max_tokens):
    llm=Ollama(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## #Title of the app
st.title("Enhanced Q&A Chatbot With Open Source Models")
st.markdown("""
            You can use various types of **Ollama models from the github [https://github.com/ollama/ollama].**
            By visiting the Ollama site and downloading the ollama app you can go to the documentation part and 
            see the various types of the models you can use. 
            """)

## Select the OpenAI model
llm=st.sidebar.selectbox("Select Open Source model",["mistral","Gemma 3","Llama 3.1"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if st.button("Enter") and user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
elif st.button("Enter"):
    st.write("Please provide the user input")


