from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

st.header('Simple Static Chatbot ')

prompt = st.chat_input("Say something")
if prompt:
    response=model.invoke(prompt)
    st.write(f"User: {prompt}")
    st.write(f"Model: {response.content[0]['text']}")