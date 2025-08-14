import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings  # noqa: F401 
# Ollama Embeddings take a lot of time to convert the words into the embeddings so in that chase we use the openai embeddings technique as it convert the word to the vectors usings the api keys and that would be very fast 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain #This is basically used to create the Retrieval-Augmented Generation (RAG) 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain # important for RAG and any Q&A Application
from langchain_community.vectorstores import FAISS # this is the vector store database we can also use the chroma database  
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings 
import time
import openai  # noqa: F401
from dotenv import load_dotenv
load_dotenv()

"""
it's a technique that enhances the capabilities of language models by combining them with external knowledge sources.
Essentially, RAG allows a language model to access and incorporate information from databases or other documents 
when generating responses, making its answers more accurate, relevant, and up-to-date. 

"""
## load the GROQ API Key
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY") 
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY") # noqa: F401

## If you do not have open AI key use the below Huggingface embedding # noqa: F401
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN") 
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192") # noqa: F401

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

def create_vector_embedding():
    if "vectors" not in st.session_state: # That are Chat would have a memory related to our chatbot  
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading state 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)#Splitter documents
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
st.title("RAG Document Q&A With Groq And Lama3")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)#Create a chain for passing a list of Documents to a model.
    retriever=st.session_state.vectors.as_retriever() # if we want to retrieve any data from my vector db then i had to first convert that into a retriever
    #retriever act as the interface to pass query to your vector database and in this case we have used the FAISS database 
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"): # this returns the same-same meanings of the text where it is used in the research paper and bring the similar
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')






