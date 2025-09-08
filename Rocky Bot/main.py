import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import validators
import os 
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RockyBot",page_icon=" ")
st.title("Rocky Chatbot For Reading Any URL's")
st.subheader("Summarize Any URl's and asky any thing about them with this rocky bot")
st.sidebar.title("Give the URL's Link Here")
urls=[]
for i in range(3):
  generic_url=st.sidebar.text_input(f"URL {i + 1} from which you want to read")
  urls.append(generic_url)

process_url_clicked=st.sidebar.button("Process The URL's")

llm=ChatGroq(model="llama-3.1-8b-instant",api_key=os.getenv("GROQ_API_KEY"))

prompt_template="""
  You are a helful chatbot named RockyBot created by Dhruv Agarwal the user will provide you the Website URL's you need to thorougly read the URL Link content
  and answer the queries of the user input if mentions about the size from the users see acoording otherwise give answers in about 50-60 words
  Content:{text}
  """
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

#processing pipeline 
if process_url_clicked:
  valid_urls = [u for u in urls if validators.url(u)]
  if not valid_urls:
    st.error("Please enter at least one valid URL ...")
  else:
    st.write("Processing URL's ...")
    Loader=UnstructuredURLLoader(urls=urls)
    data=Loader.load()
    st.write("Data is Loaded Succesfully ...")
    text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=1000,chunk_overlap=0)
    docs=text_splitter.split_documents(data)
    st.write("Data is splitted into chunks ...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.write("Storing the Chunks in Vectordb ...")
    retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
    st.write("Retrieval Done ...")

    user_query=st.text_input("Ask Any question you need to ask about the URL's")

    if user_query:
      if "summary" in user_query.lower():
        chain=load_summarize_chain(llm,chain_type="map_reduce",map_prompt=prompt,combine_prompt=prompt,verbose=True)
        summary = chain.run(docs)
        st.subheader("📌 Summary:")
        st.write(summary)
      else:
        chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=True)
        result=chain.run(user_query)
        st.write("📌Answer:")
        st.write(result)

