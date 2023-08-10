from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st
from streamlit_chat import message
import os
import base64
import time
import torch
from components.sidebar import sidebar

DATA_PATH = 'data/'

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = DirectoryLoader(DATA_PATH,
                                         glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    #create vector store here
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


#Retrieval QA Chain
#@st.cache_resource
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
#@st.cache_resource
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.01
    )
    return llm

#QA Model Function
#@st.cache_resource
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

def process_answer(instruction):
    response = '' 
    instruction = instruction
    qa = qa_bot()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))


# streamlit code
def main():
    st.set_page_config(page_title="PrivateGPT", page_icon="📖", layout="wide")
    st.header("📖PrivateGPT")

    sidebar()
    

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
       
        filepath = "data/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
                st.success('Embeddings are created successfully!')
                st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)


        user_input = st.text_input("", key="input")

        # Initialize session state for generated responses and past messages
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]

       # Search the database for a response based on user input and update session state
        if user_input:
            answer = process_answer({'query': user_input})
            st.session_state["past"].append(user_input)
            response = answer
            st.session_state["generated"].append(response)

        # Display conversation history using Streamlit messages
        if st.session_state["generated"]:
            display_conversation(st.session_state)


if __name__ == "__main__":
    main()                                                                                                                                                                                                                                                                                           
