
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] #streamlit secrets
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"] #streamlit secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"] #streamlit secrets


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain_projects"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


#Prompt Template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond to the user's question."),
    ("user", "Question:{question}") ])


#Streamlit Framework
st.title("Langchain Projects")
input_text = st.text_input("Enter your question here:")

#openai LLM

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
llm_lama2 = Ollama(model="llama2")
llm_huggingface = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

#alternative usage of HuggingFacePipeline
# llm = HuggingFacePipeline.from_model_id(
#     model_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )

output_parser=StrOutputParser()

chain = prompt | llm_huggingface | output_parser

if input_text:
    output = chain.invoke({"question":input_text})
    st.write(output)