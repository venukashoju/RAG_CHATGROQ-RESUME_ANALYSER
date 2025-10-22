import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
# groq_api_key=os.getenv("GROQ_API_KEY")


groq_api_key=st.sidebar.text_input("Enter Your Groq API key",type='password')
llm = ChatGroq(groq_api_key=groq_api_key,model_name="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_template(
    """
    You are Resume analyser. You analyses resume based on user question and 
    gives pros,cons and ATS score for resume. Edit the given resume to ATS friendly.
    Give 2 to 4 edited resume samples.
    
    Please provide the most accurate response based on the question.
    Present the result in professional way.
    <context>
    {context}
    <context>
    Question:{input}

    """
)


st.title("ATS Friendly Resume Analyser")
uploaded_file = st.sidebar.file_uploader("Upload Your Resume",type='pdf')
if uploaded_file is not None:
    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.read())


def create_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        st.session_state.loader=PyPDFLoader('resume.pdf')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
        st.session_state.final_doc=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_doc,st.session_state.embeddings)


if st.sidebar.button("Document Embedding"):
    create_embeddings()
    st.sidebar.write("Vector Database is ready")

user_prompt = st.sidebar.text_input("Enter your query")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever= st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retriever_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response["answer"])

    with st.expander("Document Similariy Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
