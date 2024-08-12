import streamlit as st
import requests

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/openai/invoke", json={"input":{"topic": input_text}})
    return response.json()

def get_llama2_response(input_text):
    response = requests.post("http://localhost:8000/llama2/invoke", json={"input":{"topic": input_text}})
    return response.json()

st.title("Langchain Projects")
input_text_openai = st.text_input("Write a story on the topic:")
input_text_llama2 = st.text_input("Write a poem on the topic:")

if input_text_openai:
    output_openai = get_openai_response(input_text_openai)
    st.write(output_openai)
    
if input_text_llama2:
    output_llama2 = get_llama2_response(input_text_llama2)
    st.write(output_llama2)