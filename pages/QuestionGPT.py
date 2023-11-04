import streamlit as st 
from Home import answer_prompt




st.title('🦜🔗 Resume GPT - Question Answering Assistant')

job_desc = st.text_area('Enter Job Description')

question = st.text_input('Enter hiring question')

run = st.button('Run')

if run:
    prompt = 'Answer this question using information about Elias and this job description \n Question: ' + question + ' Job description: ' + job_desc

    response = answer_prompt(prompt)

    st.write(response)