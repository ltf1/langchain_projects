
from fastapi import FastAPI, Request

from langserve import add_routes
import uvicorn
import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



app = FastAPI(
    title = "Langchain Projects",
    version = "0.1.0",
    description =  "This is a simple API that uses Langchain to answer questions about Langchain Projects"
)

#Prompt Template

prompt_openai = ChatPromptTemplate.from_template("You are a helpful assistant. Write a short story about the {topic}")

prompt_llama2 = ChatPromptTemplate.from_template("You are a helpful assistant. Write a short poem about the {topic}")

#openai LLM

llm_openai = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
llm_llama2 = Ollama(model="llama2")

output_parser=StrOutputParser()

chain_openai = prompt_openai | llm_openai | output_parser
chain_llama = prompt_llama2 | llm_llama2 | output_parser


add_routes(
    app, 
    chain_openai,
    path="/openai")

add_routes(
    app,
    chain_llama,
    path = "/llama2")

import logging

logging.basicConfig(level=logging.DEBUG)

@app.post("/openai/invoke")
async def openai_invoke(request: Request):
    data = await request.json()
    logging.debug(f"Received data: {data}")
    response = chain_openai(data)
    logging.debug(f"OpenAI response: {response}")
    return response

@app.post("/llama2/invoke")
async def llama2_invoke(request: Request):
    data = await request.json()
    logging.debug(f"Received data: {data}")
    response = chain_llama(data)
    logging.debug(f"Llama2 response: {response}")
    return response




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)