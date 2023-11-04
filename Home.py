from getpass import getpass
import streamlit as st 
import os 
from langchain import OpenAI
#import pdf document loaders
from langchain.document_loaders import PyPDFLoader 
#chroma for vector store
from langchain.vectorstores import Chroma
#impect vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from dotenv import load_dotenv


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')


llm = OpenAI(temperature=0.9, verbose=True)

embeddings = OpenAIEmbeddings()


loader = PyPDFLoader('elias_dzobo.pdf')

#split pages
pages = loader.load_and_split()

#load pages into vector database; ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='elias_dzobo')

vectorstore_info = VectorStoreInfo(
    name='elias_dzobo',
    description='Elias Dzobo Resume',
    vectorstore = store 
)

#convert document into langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.title('🦜🔗 Resume GPT')
prompt = st.text_input('Input your prompt here')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)


col1, col2, col3, col4 = st.columns(4)

with col1:
    info = st.button('info')

with col2:
    work = st.button('work experience')

with col3:
    projects = st.button('projects')

with col4:
    skill = st.button('skillset')


if info:
    prompt = 'Write a brief introduction about Elias Dzobo'
    response = agent_executor.run(prompt)
    st.write(response)

if work:
    prompt = "Write a detailed exert on Elias Dzobo's work experience highlighting his accomplishments and skills learnt"
    response = agent_executor.run(prompt)
    st.write(response)

if projects:
    prompt = "Write a detailed exert on project Elias Dzobo has worked on, highlighting his accomplishments and skills learnt"
    response = agent_executor.run(prompt)
    st.write(response)

if skill:
    prompt = "Write a detailed exert on Elias Dzobo's skillset"
    response = agent_executor.run(prompt)
    st.write(response)


st.subheader('Job Fit')

job_desc = st.text_area('Job Description')
fit = st.button('Job Fit')

job_template = PromptTemplate(
    input_variables = ['Job'],
    template = 'Is Elias a good candidate for the following role {Job}'
)

prompt = 'Is Elias a good candidate for the following role: \n' + job_desc


#job_chain  = LLMChain(llm=agent_executor, prompt=job_template)
if fit and job_desc:
    response = agent_executor.run(prompt)
    st.write(response) 



def answer_prompt(prompt):
    response = agent_executor.run(prompt)
    return response