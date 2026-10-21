from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGroq(model='qwen/qwen3.8-27b')

print(model.invoke("what is the capital of India").content)
